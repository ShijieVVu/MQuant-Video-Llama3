import math
import time
import tqdm
import torch
import torch.nn as nn
from fake_quant import utils
from fake_quant import quant_utils
from fake_quant.gptq.gptq_utils import GPTQ, GPTQConv
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def videollama3_visual_encoder_rtn(model, dev, args, quantizers):
    """
    RTN quantization for VideoLLaMA3 vision encoder
    Similar to Qwen2VL but adapted for VideoLLaMA3's structure
    """
    # Vision encoder patch embedding
    print("-----RTN Quantization vision encoder patch embedding-----")
    vision_encoder = model.vision_encoder
    
    # Check if patch embedding exists
    if hasattr(vision_encoder, 'embeddings') and hasattr(vision_encoder.embeddings, 'patch_embedding'):
        quantizer = quant_utils.WeightQuantizer()
        quantizer.configure(
            args.visual_w_bits,
            perchannel=True,
            sym=not (args.w_asym),
            mse=args.visual_w_clip,
        )
        W = vision_encoder.embeddings.patch_embedding.weight.data
        quantizer.find_params(W)
        vision_encoder.embeddings.patch_embedding.weight.data = quantizer.quantize(W).to(
            vision_encoder.embeddings.patch_embedding.weight.dtype
        )
        quantizers["model.vision_encoder.embeddings.patch_embedding"] = quantizer.cpu()

    # Vision encoder transformer layers
    layers = vision_encoder.encoder.layers
    for i in tqdm.tqdm(
        range(len(layers)), desc="(RtN Quant.) vision encoder Layers"
    ):
        layer = layers[i]

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            if any(p_name in name for p_name in args.skip_names):
                continue
            layer_weight_bits = args.visual_w_bits
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.visual_w_clip,
            )
            W = subset[name].weight.data
            dtype = W.dtype
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(dtype)
            quantizers["model.vision_encoder.encoder.layers.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_videollama3_fwrd_visual_encoder_patch_embedding(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    GPTQ for VideoLLaMA3 vision encoder patch embedding
    """
    print("-----GPTQ Quantization vision encoder patch embedding-----")
    inps = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, pixel_values, **kwargs):
            inps[cache["i"]] = pixel_values
            cache["i"] += 1
            raise ValueError

    vision_encoder = model.model.vision_encoder
    vision_encoder.embeddings = Catcher(vision_encoder.embeddings)

    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break

        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass

    vision_encoder.embeddings = vision_encoder.embeddings.module
    
    # Quantize patch embedding if it exists
    if hasattr(vision_encoder.embeddings, 'patch_embedding'):
        layer_weight_bits = args.visual_w_bits
        layer_weight_sym = not (args.w_asym)
        
        patch_gptq = GPTQ(vision_encoder.embeddings.patch_embedding)
        patch_gptq.quantizer = quant_utils.WeightQuantizer()
        patch_gptq.quantizer.configure(
            layer_weight_bits,
            perchannel=True,
            sym=layer_weight_sym,
            mse=args.visual_w_clip,
        )

        def add_batch():
            def tmp(_, inp, out):
                patch_gptq.add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        handles.append(
            vision_encoder.embeddings.patch_embedding.register_forward_hook(add_batch())
        )
        
        for j in range(args.nsamples):
            vision_encoder.embeddings(inps[j])
            
        for h in handles:
            h.remove()
            
        layer_w_groupsize = args.w_groupsize
        patch_gptq.fasterquant(
            percdamp=args.percdamp,
            groupsize=layer_w_groupsize,
            actorder=args.act_order,
            static_groups=False,
        )
        quantizers["model.vision_encoder.embeddings.patch_embedding"] = patch_gptq.quantizer.cpu()
        patch_gptq.free()
        del patch_gptq
        
    utils.cleanup_memory(verbos=True)
    print("-----GPTQ Quantization vision encoder patch embedding Done-----")


@torch.no_grad()
def gptq_videollama3_fwrd_visual_encoder_layers(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    GPTQ for VideoLLaMA3 vision encoder transformer layers
    """
    print("-----GPTQ Quantization vision encoder layers-----")
    vision_encoder = model.model.vision_encoder
    layers = vision_encoder.encoder.layers

    inps = [None] * args.nsamples
    grid_sizes = [None] * args.nsamples
    merge_sizes = [None] * args.nsamples

    cache = {"i": 0}

    # Monkey patch to capture inputs
    def monkey_patched_forward(self, *args, **kwargs):
        inps[cache["i"]] = args[0]  # Store input
        grid_sizes[cache["i"]] = kwargs.get("grid_sizes", None)
        merge_sizes[cache["i"]] = kwargs.get("merge_sizes", None)
        cache["i"] += 1
        raise ValueError("Catcher triggered")

    # Apply monkey patch to first layer
    forward = layers[0].forward
    layers[0].forward = monkey_patched_forward.__get__(layers[0], layers[0].__class__)

    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass

    layers[0].forward = forward

    outs = [None] * args.nsamples

    # Sequential processing for attention and MLP
    sequential = [
        ["self_attn.qkv"],
        ["self_attn.proj"],
        ["mlp.fc1"],
        ["mlp.fc2"],
    ]

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i]
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        
        for names in sequential:
            if any(p_name in name for p_name in args.skip_names):
                continue
            subset = {n: full[n] for n in names if n in full}
            
            if not subset:
                continue

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.visual_w_bits
                layer_weight_sym = not (args.w_asym)
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.visual_w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(args.nsamples):
                kwargs = {}
                if grid_sizes[j] is not None:
                    kwargs["grid_sizes"] = grid_sizes[j]
                if merge_sizes[j] is not None:
                    kwargs["merge_sizes"] = merge_sizes[j]
                outs[j] = layer(inps[j], **kwargs)
                
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers["model.vision_encoder.encoder.layers.%d.%s" % (i, name)] = gptq[
                    name
                ].quantizer.cpu()
                gptq[name].free()

        # Final forward pass for this layer
        for j in range(args.nsamples):
            kwargs = {}
            if grid_sizes[j] is not None:
                kwargs["grid_sizes"] = grid_sizes[j]
            if merge_sizes[j] is not None:
                kwargs["merge_sizes"] = merge_sizes[j]
            outs[j] = layer(inps[j], **kwargs)

        layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization vision encoder layers Done----")
    return quantizers


def videollama3_visual_projector_rtn(model, dev, args, quantizers):
    """
    RTN quantization for VideoLLaMA3 vision projector (mm_projector)
    """
    print("-----RTN Quantization vision projector-----")
    mm_projector = model.mm_projector
    
    subset = quant_utils.find_qlayers(mm_projector, layers=[torch.nn.Linear])
    for name in subset:
        if any(p_name in name for p_name in args.skip_names):
            continue
        layer_weight_bits = args.visual_w_bits
        quantizer = quant_utils.WeightQuantizer()
        quantizer.configure(
            layer_weight_bits,
            perchannel=True,
            sym=not (args.w_asym),
            mse=args.visual_w_clip,
        )
        W = subset[name].weight.data
        quantizer.find_params(W)
        subset[name].weight.data = quantizer.quantize(W).to(subset[name].weight.dtype)
        quantizers["model.mm_projector.%s" % name] = quantizer.cpu()


@torch.no_grad()
def gptq_videollama3_fwrd_visual_projector(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    GPTQ for VideoLLaMA3 vision projector
    """
    print("-----GPTQ Quantization vision projector-----")
    inps = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            inps[cache["i"]] = args[0]
            cache["i"] += 1
            raise ValueError

    # Wrap mm_projector
    original_projector = model.model.mm_projector
    model.model.mm_projector = Catcher(model.model.mm_projector)
    
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
            
    model.model.mm_projector = model.model.mm_projector.module

    # Quantize projector layers
    full = quant_utils.find_qlayers(model.model.mm_projector, layers=[torch.nn.Linear])
    
    # For linear projector or MLP projector
    if isinstance(model.model.mm_projector, nn.Linear):
        # Single linear layer
        sequential = [["weight"]]
        layer_names = [""]
    else:
        # MLP projector - typically has multiple layers
        sequential = []
        layer_names = []
        for name in full:
            sequential.append([name])
            layer_names.append(name)

    for names in sequential:
        subset = {n: full[n] for n in names if n in full}
        
        if not subset:
            continue

        gptq = {}
        for name in subset:
            print(f"{name}", end="  ", flush=True)
            layer_weight_bits = args.visual_w_bits
            layer_weight_sym = not (args.w_asym)
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = quant_utils.WeightQuantizer()
            gptq[name].quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=layer_weight_sym,
                mse=args.visual_w_clip,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        for j in range(args.nsamples):
            model.model.mm_projector(inps[j])
            
        for h in handles:
            h.remove()

        for name in subset:
            layer_w_groupsize = args.w_groupsize
            gptq[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=layer_w_groupsize,
                actorder=args.act_order,
                static_groups=False,
            )
            quantizers["model.mm_projector.%s" % name] = gptq[name].quantizer.cpu()
            gptq[name].free()

    del gptq
    torch.cuda.empty_cache()
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization vision projector Done-----")


def videollama3_llm_rtn(model, dev, args, quantizers):
    """
    RTN quantization for VideoLLaMA3 LLM (Qwen2-based)
    """
    print("-----RTN Quantization LLM-----")
    layers = model.model.layers
    torch.cuda.empty_cache()

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) LLM Layers"):
        layer = layers[i]

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            if any(p_name in name for p_name in args.skip_names) or "L1" in name:
                continue
            layer_weight_bits = args.llm_w_bits
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.llm_w_clip,
            )
            W = subset[name].weight.data
            dtype = W.dtype
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(dtype)
            quantizers["model.model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_videollama3_fwrd_llm(model, dataset, dev, dataset_name, args, quantizers):
    """
    GPTQ for VideoLLaMA3 LLM (Qwen2-based language model)
    Adapted from Qwen2VL GPTQ implementation
    """
    print("-----GPTQ Quantization LLM-----")
    use_cache = model.model.config.use_cache
    model.model.config.use_cache = False
    
    # Access the Qwen2 layers
    layers = model.model.model.layers

    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    position_ids = [None] * args.nsamples
    cache_position = [None] * args.nsamples

    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.register_module('module', module)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            attention_masks[cache["i"]] = kwargs.get("attention_mask", None)
            position_ids[cache["i"]] = kwargs.get("position_ids", None)
            cache_position[cache["i"]] = kwargs.get("cache_position", None)
            cache["i"] += 1
            raise ValueError
        
        def __getattr__(self, name):
            if name == 'module':
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            try:
                module = object.__getattribute__(self, '_modules')['module']
                return getattr(module, name)
            except (AttributeError, KeyError):
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    layers[0] = Catcher(layers[0])
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt)):
        if cache["i"] >= args.nsamples:
            break
        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(
            dataset_name
        ):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except ValueError:
            pass
            
    layers[0] = layers[0].module

    outs = [None] * args.nsamples

    # Sequential processing for Qwen2 layers
    sequential = [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
    ]
    
    if args.llm_split:
        sequential.append(["mlp.down_proj.L2"])
    else:
        sequential.append(["mlp.down_proj"])
        
    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i]
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        
        for names in sequential:
            if any(p_name in name for p_name in args.skip_names):
                continue
            subset = {n: full[n] for n in names if n in full}
            
            if not subset:
                continue

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.llm_w_bits
                layer_weight_sym = not (args.w_asym)
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.llm_w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
                
            for j in range(args.nsamples):
                kwargs = {}
                if attention_masks[j] is not None:
                    kwargs["attention_mask"] = attention_masks[j]
                if position_ids[j] is not None:
                    kwargs["position_ids"] = position_ids[j]
                if cache_position[j] is not None:
                    kwargs["cache_position"] = cache_position[j]
                outs[j] = layer(inps[j], **kwargs)[0]
                
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers["model.model.layers.%d.%s" % (i, name)] = gptq[
                    name
                ].quantizer.cpu()
                gptq[name].free()

        # Final forward pass for this layer
        for j in range(args.nsamples):
            kwargs = {}
            if attention_masks[j] is not None:
                kwargs["attention_mask"] = attention_masks[j]
            if position_ids[j] is not None:
                kwargs["position_ids"] = position_ids[j]
            if cache_position[j] is not None:
                kwargs["cache_position"] = cache_position[j]
            outs[j] = layer(inps[j], **kwargs)[0]

        layers[i] = layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    print("\n-----GPTQ Quantization LLM Done-----")
    return quantizers


@torch.no_grad()
def videollama3_rtn_gptq_fwrd_plus(model, dataset, dev, dataset_name, args):
    """
    Main function for VideoLLaMA3 quantization
    Supports both RTN and GPTQ for vision encoder, projector, and LLM
    """
    logging.info("-----RTN Or GPTQ Quantization for VideoLLaMA3-----")

    quantizers = dict()

    # Quantize vision encoder
    if args.quant_visual_clip:
        if args.visual_w_rtn:
            videollama3_visual_encoder_rtn(model.model, dev, args, quantizers)
        else:
            gptq_videollama3_fwrd_visual_encoder_patch_embedding(
                model, dataset, dev, dataset_name, args, quantizers
            )
            gptq_videollama3_fwrd_visual_encoder_layers(
                model, dataset, dev, dataset_name, args, quantizers
            )

    # Quantize vision projector
    if args.quant_visual_projector:
        if args.visual_w_rtn:
            videollama3_visual_projector_rtn(model.model, dev, args, quantizers)
        else:
            gptq_videollama3_fwrd_visual_projector(
                model, dataset, dev, dataset_name, args, quantizers
            )

    # Quantize LLM
    if args.quant_llm:
        if args.llm_w_rtn:
            videollama3_llm_rtn(model.model, dev, args, quantizers)
        else:
            gptq_videollama3_fwrd_llm(model, dataset, dev, dataset_name, args, quantizers)
            
    return quantizers

