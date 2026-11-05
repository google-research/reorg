# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ReOrg: a single worker inside an organizational reflection layer.

    reflection_skill/    organizational reflection skills and one reflection round   Sec. 3.3, Eq. 1
        graph.py         a skill s = (V, E): roles, actions, validation, fingerprint
        reflect.py       Reflect, then the worker's revision
        chronicle.py     the work chronicle H_r
        loop_in.py       the Loop-In protocol
    continual_learning/  learning the library and planner knowledge             Sec. 4, App. A.2-A.3
        learn.py         Alg. 1 over the adaptation stream
        seeds.py, search.py, mutate.py, ranking.py, aggregate.py, knowledge.py, later_round.py
    planner.py           first-round and later-round retrieval, completion gate   App. A.3
    inference.py         up to three reflection rounds with a frozen (L, M)
    library.py           the skill library L and planner knowledge M
    worker.py            Stirrup sessions for the worker, stakeholders and judges
    llm.py               Gemini Developer API and native vLLM clients
    web_search.py        the worker's web search
    domain.py            the benchmark interface: task packets x = (u, R) and task sets
    prompts/             the prompt templates of App. A.6-A.7 and their loader
    files.py             ID lists, hashing and JSON helpers
"""

__version__ = "0.1.0"
