import torch, torch.nn as nn, torch.nn.functional as F, argparse, datetime, os
from datasets import load_dataset
from loguru import logger
from evaluation.eval import eval_dataset
from fake_quant import quant_utils
from fake_quant import gptq
import functools
from fake_quant import utils
from fake_quant import hadamard_utils
from vlmeval.config import supported_VLM
from vlmeval.dataset.image_vqa import OCRBench
from vlmeval.smp import load, osp

torch.set_grad_enabled(False)

def init_logger(args):
    logger_file = str(datetime.datetime.now().strftime("%m-%d %H:%M:%S")) + ".log"
    os.makedirs("log", exist_ok=True)
    if args.model_name is not None:
        logger_file = args.model_name + "_" + logger_file
    logger_file = "log/" + logger_file
    logger.add(logger_file)

Model_Setting = {
    "VideoLLaMA3-7B": "DAMO-NLP-SG/VideoLLaMA3-7B",
    "VideoLLaMA3-13B": "DAMO-NLP-SG/VideoLLaMA3-13B",
}

class LocalOCRBench(OCRBench):
    """Custom OCRBench dataset that loads from a local TSV file."""
    def __init__(self, dataset="OCRBench", local_file=None, **kwargs):
        self.local_file = local_file
        # Call parent __init__ which will call our overridden load_data
        super().__init__(dataset=dataset, **kwargs)
    
    def load_data(self, dataset):
        if self.local_file and osp.exists(self.local_file):
            # Load directly from local file
            data = load(self.local_file)
            # Keep header (row 0) and first 2 data rows (rows 1-2)
            # This means indices 0, 1, 2 (header + 2 data rows)
            if len(data) > 2:
                data = data.iloc[:3]  # Header + 2 data rows
            return data
        else:
            # Fall back to default OCRBench behavior
            return super().load_data(dataset)

def main(args):
    model_name = args.model_name
    model = supported_VLM[model_name](
        model_path=Model_Setting[model_name], verbose=args.verbose
    )

    utils.seed_everything(args.seed)
    
    # Note: VideoLLaMA3 uses different fusion/rotation than Qwen2VL
    # Adapt layer norm fusion if needed (placeholder for now)
    # if not args.not_fuse_layer_norms:
    #     fuse_videollama3_layer_norms(model, args)
    
    # Rotation for VideoLLaMA3 (if rotation is implemented)
    # if args.rotate:
    #     rotate_videollama3_model(model.model, args)

    # Online Hadamard for LLM
    if not args.quant and args.online_llm_hadamard:
        if args.rotate_llm:
            args.quant_llm = True
        quant_utils.videollama3_add_act_quant(model, args)
        qlayers = quant_utils.find_qlayers(
            model.model.model, layers=[quant_utils.ActQuantWrapper]
        )
        for name in qlayers:
            if "mlp.down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
                if hasattr(model.model.config, 'need_pad') and model.model.config.need_pad:
                    hook = functools.partial(
                        utils.revise_down_input,
                        new_size=model.model.config.intermediate_size,
                    )
                    qlayers[name].register_forward_pre_hook(hook)

    # Online Hadamard for Visual Encoder
    if not args.quant and args.online_visual_hadamard:
        if args.rotate_visual_encoder:
            args.quant_visual_encoder = True
        quant_utils.videollama3_add_act_quant_visual(model, args)
        
        if args.rotate_visual_encoder and args.visual_split:
            print("adding online hadamard rotation for vision encoder")
            qlayers = quant_utils.find_qlayers(
                model.model.vision_encoder, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if "mlp.fc2" in name:
                    # Get the feature dimension from the first layer
                    had_K, K = hadamard_utils.get_hadK(
                        int(model.model.vision_encoder.encoder.layers[0].mlp.fc2.in_features)
                    )
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    qlayers[name].split = args.visual_split
                    if args.visual_split:
                        qlayers[name].split_weights()

    if args.quant:
        if args.load_gptq:
            print("Loading GPTQ model from: ", args.load_gptq)
            model.model = torch.load(args.load_gptq)
        else:
            # from torch.utils.data import ConcatDataset
            from vlmeval.dataset import build_dataset

            # Use local dataset file if provided
            if args.local_dataset_file:
                dataset = LocalOCRBench(dataset=args.dataset_name, local_file=args.local_dataset_file)
            else:
                dataset = build_dataset(args.dataset_name)

            # Use VideoLLaMA3 quantization function
            quantizers = gptq.videollama3_rtn_gptq_fwrd_plus(
                model, dataset, utils.DEV, args.dataset_name, args
            )
            if args.dump_gptq:
                torch.save(model.model, args.dump_gptq)
                print("Dumped the GPTQ model to: ", args.dump_gptq)

        # Configure activation quantization for vision encoder
        if args.visual_a_bits < 16 or args.visual_static:
            if args.visual_static and args.visual_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            qlayers = quant_utils.find_qlayers(
                model.model.vision_encoder, layers=[quant_utils.ActQuantWrapper]
            )

            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.visual_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.visual_static,
                    observer_type="minmax",
                )

        # Configure activation quantization for LLM
        if args.llm_a_bits < 16 or args.llm_static:
            if args.llm_static and args.llm_a_bits >= 16:
                print("if you want to run act with fp16, please set --static False")
            qlayers = quant_utils.find_qlayers(
                model.model.model, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                layer_input_bits = args.llm_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.llm_static,
                    observer_type="minmax",
                )

    from vlmeval.dataset import build_dataset

    # Use local dataset file if provided
    if args.local_dataset_file:
        dataset = LocalOCRBench(dataset=args.dataset_name, local_file=args.local_dataset_file)
    else:
        dataset = build_dataset(args.dataset_name)


    # Calibration for static quantization
    if args.llm_static or args.visual_static:
        quant_utils.calib_videollama3_plus(model, args, dataset, args.calib_num)
    
    if args.max_new_tokens is not None:
        model.kwargs["max_new_tokens"] = args.max_new_tokens
    
    # Evaluation
    eval_dataset(
        model,
        dataset,
        args.dataset_name,
        model_name=model_name,
        verbose=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="VideoLLaMA3-7B")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--quant", action="store_true")

    # Rotation Arguments
    parser.add_argument(
        "--rotate", action="store_true", default=False, help="""Rotate the model. """
    )
    parser.add_argument(
        "--analysis", action="store_true", default=False, help="""analysis act. """
    )
    parser.add_argument(
        "--analysis_c_proj",
        action="store_true",
        default=False,
        help="""analysis act. """,
    )
    parser.add_argument(
        "--draw_save_path",
        type=str,
        default="output/videollama3_base",
        help="""analysis act save path. """,
    )
    parser.add_argument(
        "--rotate_visual_encoder",
        action="store_true",
        default=False,
        help="""Rotate the vision encoder. """,
    )
    parser.add_argument(
        "--rotate_visual_projector",
        action="store_true",
        default=False,
        help="""Rotate the visual projector. """,
    )
    parser.add_argument(
        "--rotate_llm",
        action="store_true",
        default=False,
        help="""Rotate the LLM. """,
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--visual_a_bits",
        type=int,
        default=8,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    # Activation Quantization Arguments
    parser.add_argument(
        "--llm_a_bits",
        type=int,
        default=8,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action="store_true",
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--visual_w_bits",
        type=int,
        default=4,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--llm_w_bits",
        type=int,
        default=4,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action="store_true",
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--visual_w_rtn",
        action="store_true",
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--llm_w_rtn",
        action="store_true",
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--visual_w_clip",
        action="store_true",
        default=False,
        help="""Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--llm_w_clip",
        action="store_true",
        default=False,
        help="""Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order", action="store_true", default=False, help="act-order in GPTQ"
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action="store_true",
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    parser.add_argument(
        "--quant_llm",
        action="store_true",
        default=False,
        help="Quantize the VideoLLaMA3 LLM model",
    )

    parser.add_argument(
        "--quant_visual_clip",
        action="store_true",
        default=False,
        help="Quantize the vision encoder (clip) model",
    )

    parser.add_argument(
        "--quant_visual_projector",
        action="store_true",
        default=False,
        help="Quantize the visual projector (mm_projector)",
    )

    parser.add_argument(
        "--act_per_tensor",
        action="store_true",
        default=False,
        help="Quantize the activations per tensor",
    )

    parser.add_argument(
        "--nsamples",
        type=int,
        default=8,
        help="Number of calibration data samples for GPTQ.",
    )

    parser.add_argument(
        "--skip_names",
        nargs="+",
        default=[],
        help="Skip the quantization of the layers with these names",
    )

    parser.add_argument(
        "--no_fuse_visual_encoder",
        action="store_true",
        default=False,
        help="Don't fuse vision encoder layer norms",
    )

    parser.add_argument(
        "--no_fuse_visual_projector",
        action="store_true",
        default=False,
        help="Don't fuse visual projector",
    )

    parser.add_argument(
        "--no_fuse_llm",
        action="store_true",
        default=False,
        help="Don't fuse LLM layer norms",
    )
    parser.add_argument(
        "--not_fuse_layer_norms",
        action="store_true",
        default=False,
        help="Don't fuse any layer norms",
    )
    parser.add_argument(
        "--llm_static",
        action="store_true",
        default=False,
        help="quant act with static scale and zero point",
    )

    parser.add_argument(
        "--visual_static",
        action="store_true",
        default=False,
        help="quant act with static scale and zero point",
    )

    parser.add_argument(
        "--calib_num",
        type=int,
        default=32,
        help="calibration number",
    )

    parser.add_argument(
        "--eval_num",
        type=int,
        default=32,
        help="evaluation number",
    )

    parser.add_argument(
        "--calib_mode",
        type=str,
        default="v2",
        help="calibration mode, v1 or v2",
    )

    parser.add_argument(
        "--analysis_num",
        type=int,
        default=32,
        help="analysis number",
    )

    parser.add_argument(
        "--analysis_mode",
        type=str,
        default="v1",
        help="analysis mode, v1 or v2",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="TextVQA_VAL",
        help="dataset name",
    )
    parser.add_argument(
        "--local_dataset_file",
        type=str,
        default=None,
        help="Path to local TSV file to use instead of downloading dataset",
    )
    parser.add_argument(
        "--analysis_text",
        action="store_true",
        default=False,
        help="analysis text",
    )
    parser.add_argument(
        "--online_visual_hadamard",
        action="store_true",
        default=False,
        help="Online Hadamard rotation for vision encoder",
    )

    parser.add_argument(
        "--online_llm_hadamard",
        action="store_true",
        default=False,
        help="Online Hadamard rotation for LLM",
    )
    parser.add_argument(
        "--fp32_had",
        action="store_true",
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )
    parser.add_argument(
        "--dump_gptq",
        type=str,
        default=None,
        help="Dump the GPTQ model to this path",
    )
    parser.add_argument(
        "--load_gptq",
        type=str,
        default=None,
        help="Load the GPTQ model from this path",
    )
    parser.add_argument(
        "--visual_split",
        action="store_true",
        default=False,
        help="visual split for vision encoder MLP",
    )
    parser.add_argument(
        "--llm_split",
        action="store_true",
        default=False,
        help="LLM split for down projection",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="max_new_tokens",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="verbose question and output",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="use real quantization"
    )
    parser.add_argument(
        "--real_mllm",
        action="store_true",
        default=False,
        help="use real quantization for MLLM"
    )
    parser.add_argument(
        "--test_static",
        action="store_true",
        default=False,
        help="test static quantization"
    )
    parser.add_argument(
        "--test_time",
        action="store_true",
        default=False,
        help="test time measurement"
    )
    args = parser.parse_args()
    init_logger(args)
    main(args)

