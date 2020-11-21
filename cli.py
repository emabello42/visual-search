from argparse import ArgumentParser
from visualsearch import __version__
from visualsearch.models import feature_extractor  as fe
from visualsearch.repository import postgresrepo as pg_repo
from visualsearch.use_cases import save_image as uc
import visualsearch.request_objects as req
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
connection_data = {
    'dbname': 'visualsearch',
    'user': 'visualsearch',
    'password': 'visualsearch',
    'host': 'localhost'
}
feat_file = "./features.npy"

def save_images(dir_path):
    repo = pg_repo.PostgresRepo(connection_data, feat_file)
    feat_extractor= fe.FeatureExtractor()
    use_case_save_image = uc.SaveImage(repo, feat_extractor)
    request_object = req.ImageRequestObject.from_dict({'params':{'path': dir_path}})
    response = use_case_save_image.execute(request_object)
    print(response.value)

def cli(args=None):
    p = ArgumentParser(
        description="Visual Search is a service to find similar images giving an input image",
        conflict_handler='resolve'
    )
    p.add_argument(
        '-V', '--version',
        action='version',
        help='Show the conda-prefix-replacement version number and exit.',
        version="visualsearch %s" % __version__,
    )

    args = p.parse_args(args)
    save_images("/home/emmanuel/data/imagenet/")
    # do something with the args
    print("CLI template - fix me up!")

    # No return value means no error.
    # Return a value of 1 or higher to signify an error.
    # See https://docs.python.org/3/library/sys.html#sys.exit


if __name__ == '__main__':
    import sys
    cli(sys.argv[1:])
