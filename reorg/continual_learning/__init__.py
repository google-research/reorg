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
"""Continual learning of organizational knowledge (Sec. 4, Alg. 1, App. A.2-A.3).

    learn.py        Alg. 1 over the ordered adaptation stream
    seeds.py        the five seed skills
    search.py       beam search and Promote
    mutate.py       graph mutations
    ranking.py      label-free groupwise artifact ranking
    aggregate.py    the tie-aware strength fit
    knowledge.py    Summarize: retrieval lessons in M
    later_round.py  later-round precedents in M
"""
