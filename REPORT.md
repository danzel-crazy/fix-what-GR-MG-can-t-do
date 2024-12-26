environment setup:

    required pretrained model:

    ## Checkpoints
[Goal Image Generation Model](https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr_mg_release/goal_gen.ckpt)

    GR-MG/checkpoints/goal/goal_gen.ckpt 

[Multi-modal Goal Conditioned Policy](https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr_mg_release/epoch=47-step=83712.ckpt)  

    GR-MG/checkpoints/policy/epoch=47-step=83712.ckpt
[diffusion pytorch model inpainting](https://huggingface.co/gligen/gligen-inpainting-text-image-box/resolve/main/diffusion_pytorch_model.bin)  

    GR-MG/GLIGEN/diffusion_pytorch_model_inpainting.bin


how to run:

    bash ./evaluate/eval.sh  ./policy/config/train.json