import algo.d3rlpy as d3rlpy
from algo.d3rlpy.logging import TensorboardAdapterFactory
import os
import imagine_bench
from imagine_bench.utils import LlataEncoderFactory, make_d3rlpy_dataset
from evaluations import CallBack

env = imagine_bench.make('Mujoco-v0', level='rephrase')
real_data, imaginary_rollout_rephrase = env.get_dataset(level="rephrase") 
dataset = make_d3rlpy_dataset(real_data, imaginary_rollout_rephrase)

agent = d3rlpy.algos.BCConfig(
                encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            ).create(device="cuda:0")
callback = CallBack()
callback.add_eval_env(env_dict={"train": env}, eval_num=10)
agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name="mujoco",
        epoch_callback=callback.EvalCallback,
        logger_adapter=TensorboardAdapterFactory(root_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )