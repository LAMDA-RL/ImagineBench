import algo.d3rlpy as d3rlpy
import imagine_bench
from imagine_bench.utils import LlataEncoderFactory, make_d3rlpy_dataset

env = imagine_bench.make('Libero-v0', level='rephrase')
real_data, imaginary_rollout_rephrase = env.get_dataset(level="rephrase") 
dataset = make_d3rlpy_dataset(real_data, imaginary_rollout_rephrase)

agent = d3rlpy.algos.TD3PlusBCConfig(
            actor_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            critic_encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device="cuda:0")

agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name="mujoco",
    )