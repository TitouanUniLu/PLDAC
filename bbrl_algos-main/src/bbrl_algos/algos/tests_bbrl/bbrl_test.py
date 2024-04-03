from functools import partial

from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, PrintAgent

from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_algos.models.stochastic_actors import RandomContinuousOneDActor
from bbrl.agents.gymnasium import make_env, ParallelGymAgent


class BBRLTest:
    def __init__(self, n_envs):
        self.env_agent = ParallelGymAgent(
            partial(make_env, "Pendulum-v1", autoreset=True),
            n_envs,
            include_last_state=True,
            seed=1,
        )
        self.actor = RandomContinuousOneDActor()
        self.agent = TemporalAgent(Agents(self.env_agent, self.actor, PrintAgent()))

    def test_workspace_size(self, taille):
        wp = Workspace()
        self.agent(wp, n_steps=taille)
        assert wp.time_size() == taille

    def test_transition_workspace_size(self, taille):
        wp = Workspace()

        self.agent(wp, n_steps=taille)
        tr_wp = wp.get_transitions()
        print(tr_wp.variables["env/timestep"].size)
        nb_trans = tr_wp.variables["env/timestep"].size[1]
        # Un épisode dure 201 pas : du pas 0 au pas 200 inclus
        # Si la taille est inférieure ou égale à 201, il y a déjà un épisode
        nb_episodes = int((taille - 1) / 201) + 1
        assert nb_trans == taille - nb_episodes, f"real size = {nb_trans} / {taille}"

    def test_rb_size(self, taille):
        wp = Workspace()
        self.agent(wp, n_steps=taille)
        tr_wp = wp.get_transitions()
        rb = ReplayBuffer(300)
        rb.put(tr_wp)
        assert not rb.is_full, f" rb real size = {taille} / {rb.size()}"
        assert rb.size() == tr_wp.variables["env/timestep"].size[1]
        print(rb.size(), rb.is_full)


if __name__ == "__main__":
    testeur = BBRLTest(1)
    taille = 201
    testeur.test_workspace_size(taille)
    testeur.test_transition_workspace_size(taille)
    testeur.test_rb_size(taille)
