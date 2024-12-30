# cd /PATH/TO/GR_MG
export EPOCHS=(47)
export CKPT_DIR="/tmp2/danzel/GR-MG/checkpoints/policy"
# export SD_CKPT="/PATH_TO_GOAL_GEN_MODEL_CKPT/epoch=49-step=51450.ckpt"
export SD_CKPT="/tmp2/danzel/GR-MG/checkpoints/goal/goal_gen.ckpt"
export MESA_GL_VERSION_OVERRIDE=3.3
echo $EPOCHS
echo $CKPT_DIR
# sudo chmod 777 -R ${CKPT_DIR}

export COUNTER=0
# Use a for loop to iterate through a list
for epoch in "${EPOCHS[@]}"; do
    export COUNTER=$((${COUNTER} + 1))
    export CUDA_VISIBLE_DEVICES=${COUNTER}
    python3 evaluate/blip_eval_cml5.py \
        --ckpt_dir ${CKPT_DIR} \
        --epoch ${epoch} \
        --ip2p_ckpt_path ${SD_CKPT} \
        --config_path ${@:1} &
done
wait