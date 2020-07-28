from setuptools import setup


setup(
    name='abyss2pano',
    version='0.1',
    description='Command line tool to stitch recordings from the abyss rig',
    url='',
    author='Andres Babino',
    author_email='ababino@gmail.com',
    license='MIT',
    packages=['abyss2pano'],
    zip_safe=False,
    scripts=['bin/abyss2pano.py'],
    install_requires=[
        'scikit-video',
        'scikit-image',
        'numpy',
        ]
    )
