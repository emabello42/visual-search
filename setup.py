from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='visual-search',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Visual Search is a service to find similar images giving an input image or text description",
    license="MIT",
    author="Emmanuel David Bello",
    author_email='emabello42@gmail.com',
    url='https://github.com/emabello42/visual-search',
    packages=['visualsearch'],
    entry_points={
        'console_scripts': [
            'visualsearch=visualsearch.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='visual-search',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
