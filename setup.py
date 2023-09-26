import setuptools

setuptools.setup(
    name="rl-exp-utils",
    version="0.16",
    author="Andrew Szot",
    author_email="andrewszot32@gmail.com",
    description="Library for RL research.",
    url="https://github.com/aszot/rl-utils",
    install_requires=[
        "matplotlib",
        "numpy>=1.16.1",
        "torch>=1.8.1",
        "subprocess32>=3.5.4",
        "gym>=0.17.1",
        "opencv-python",
        "omegaconf>=2.0.0",
        "pandas",
        "seaborn",
    ],
    packages=setuptools.find_packages(),
)
