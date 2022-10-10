import os.path as osp

import setuptools

cur_dir = osp.dirname(osp.realpath(__file__))
requirementPath = osp.join(cur_dir, "requirements.txt")
install_requires = []
with open(requirementPath) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="rl-exp-utils",
    version="0.11",
    author="Andrew Szot",
    author_email="andrewszot0@gmail.com",
    description="Library for RL research.",
    url="https://github.com/aszot/rl-utils",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
)
