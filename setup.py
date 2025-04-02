from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()


    setup(
         name = 'Cropp Recommendation model',
         version = '1.0.0',
         description = 'A model to recommend crops to farmers based on their soil conditions',
         author = 'arbaj shaikh',
         author_email = 'arbajshaikh902247@gmail.com',
    )