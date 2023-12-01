from setuptools import find_packages, setup
from glob import glob

package_name = 'final_project'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/meshes', glob('meshes/*')),
        ('share/' + package_name + '/urdf',   glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='The 133a HW6 Code',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hw6p3    = final_project.hw6p3:main',
            'hw6p4    = final_project.hw6p4:main',
            'hw6p5    = final_project.hw6p5:main',
            'racket_funcs = final_project.racket_funcs:main',
            'ball_funcs = final_project.racket_funcs:main',
            'final_project = final_project.final_project:main'
        ],
    },
)
