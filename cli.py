from argparse import ArgumentParser
from visualsearch import __version__
from visualsearch.models import feature_extractor as fe
from visualsearch.repository import postgresrepo as pg_repo
from visualsearch.use_cases import save_image as uc
from visualsearch.use_cases import find_similarities as uc_find_similarities
import visualsearch.request_objects as req
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.basicConfig(level=logging.INFO,
                    format='(%(threadName)-9s) %(message)s',)
connection_data = {
    'dbname': "visualsearchdb",
    'user': "postgres",
    'password': "",
    'host': "localhost"
}

feat_file = "./features.hdf5"


def save_images(dir_path):
    repo = pg_repo.PostgresRepo(connection_data, feat_file)
    feat_extractor= fe.FeatureExtractor()
    use_case_save_image = uc.SaveImage(repo, feat_extractor)
    request_object = req.ImageRequestObject.from_dict({'params': {'path': dir_path}})
    response = use_case_save_image.execute(request_object)
    print(response.value)


def find_similarities(file_path):
    repo = pg_repo.PostgresRepo(connection_data, feat_file, feat_mode="r")
    feat_extractor = fe.FeatureExtractor()
    use_case_find_similarities = uc_find_similarities.FindSimilarities(repo, feat_extractor)

    img = open(file_path, 'rb').read()
    request_object = req.FindSimilaritiesRequestObject.from_bytes(img)
    response = use_case_find_similarities.execute(request_object)
    logging.debug(response.value)
    rows = 3
    cols = 3
    axes = []
    fig = plt.figure()
    idx = 1
    for s in response.value:
        similar_img = mpimg.imread(s.path)
        axes.append(fig.add_subplot(rows, cols, idx))
        subplot_title = ("cosine: " + str(s.score))
        axes[-1].set_title(subplot_title)
        plt.imshow(similar_img)
        idx += 1

    fig.tight_layout()
    plt.show()



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
    # save_images("/home/emmanuel/data/imagenet/")
    find_similarities("/home/emmanuel/dev/visual-search/tests/testdata/img1.jpg")
    # do something with the args
    print("CLI template - fix me up!")

    # No return value means no error.
    # Return a value of 1 or higher to signify an error.
    # See https://docs.python.org/3/library/sys.html#sys.exit


if __name__ == '__main__':
    import sys
    cli(sys.argv[1:])
