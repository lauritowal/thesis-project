./rollout.py /Users/walter/thesis_project/checkpoints/checkpoint_111/checkpoint-111 \
--config "{\"env\": \"guidance-v0\"}"  \
--run TD3  \
--episodes 20 \
--no-render \
--jsbsim_path "/Users/walter/thesis_project/jsbsim"
#--video-dir /Users/walter/thesis_project/data/videos/