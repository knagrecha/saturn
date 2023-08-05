# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import dill
from saturn.core.executors import BaseTechnique
import os


def register(name, udp):
    """Registers a new execution technique with Saturn's library.

        Parameters:
            name: Name for the strategy to register

            udp: Class definition for your parallelism technique. Should extend the BaseTechnique class.

    """
    if not issubclass(udp, BaseTechnique):
        raise RuntimeError("""Parallelism {} is not 
                              an instance of BaseTechnique.
                              All UDPs must be subclassed 
                              from saturn.core.executors.BaseTechnique.""".format(name))

    with open("{}/{}.udp".format(os.environ["SATURN_LIBRARY_PATH"], name), 'wb') as s:
        dill.dump(udp, s)


def deregister(name):
    """Deregisters an execution technique from Saturn's library.

            Parameters:
                name: Name of the strategy to deregister

    """
    if isinstance(name, list):
        for n in name:
            os.remove("{}/{}".format(os.environ["SATURN_LIBRARY_PATH"], n))
    else:
        os.remove("{}/{}.udp".format(os.environ["SATURN_LIBRARY_PATH"], name))


def retrieve(name=None):
    """Retrieves an execution technique from Saturn's library.

                Parameters:
                    name: Name of the strategy to retrieve

    """
    if name is None:
        name = [os.path.splitext(filename)[0] for filename in os.listdir(
            os.environ["SATURN_LIBRARY_PATH"])]
    if isinstance(name, list):
        ret_cls = []
        for n in name:
            with open("{}/{}.udp".format(os.environ["SATURN_LIBRARY_PATH"], n), 'rb') as s:
                retrieved_cls = dill.load(s)
                ret_cls.append(retrieved_cls)
        return ret_cls

    else:
        with open("{}/{}.udp".format(os.environ["SATURN_LIBRARY_PATH"], name), 'rb') as s:
            retrieved_cls = dill.load(s)
            return retrieved_cls
