from langchain.chains.base import Chain
from typing import Any, List, Tuple, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import Callbacks


class FastAgent(Chain):
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
