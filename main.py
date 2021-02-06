import streamlit as st
from PIL import Image, ImageColor
import numpy as np
from sklearn.cluster import KMeans


def decompose_colors(image):

    # Decoposing image into RGB channels
    red = image[:, :, 0].reshape((-1, 1))
    green = image[:, :, 1].reshape((-1, 1))
    blue = image[:, :, 2].reshape((-1, 1))
    colors = np.hstack((red, green, blue))

    return colors


@st.cache
def cluster_colors(image, num_clusters):

    # Converting image object to numpy array
    image = np.asarray(image)
    # Decoposing image into RGB channels
    colors = decompose_colors(image)
    # Clustering colors
    cls = KMeans(num_clusters).fit(colors)
    # Predicting group colors
    clusters = cls.predict(colors)
    # Creating a color map using clusters centers
    color_map = np.array(cls.cluster_centers_).astype('uint8')
    # New image
    new_image = np.array([color_map[cluster] for cluster in clusters])
    new_image = new_image.reshape(image.shape)

    return Image.fromarray(new_image)


if __name__ == '__main__':

    st.header('Reducing Colors Using Unsupervised Learning')

    # Selector to upload image
    uploaded_image = st.sidebar.file_uploader(
        label='Choose an image', type=['png', 'jpg', 'jpeg'])

    # Selector for the number of output colors
    num_clusters = st.sidebar.number_input(
        label='Number of colors', min_value=2, max_value=10, value=4)

    # Creates two columns
    col1, col2 = st.beta_columns(2)

    # Check if an image was uploaded, otherwise loads default image
    if uploaded_image:
        image = Image.open(uploaded_image)
        with col1:
            st.image(image, caption='Original Image', use_column_width=True)
    else:
        DEFAULT_IMG = './car.jpg'
        image = Image.open(DEFAULT_IMG)
        with col1:
            st.image(image, caption='Original Image', use_column_width=True)

    # Creates the new image
    new_image = cluster_colors(image, num_clusters)

    with col2:
        st.image(new_image, caption='Transformed Image', use_column_width=True)

    """
    **How it works**
    
    The idea behind color reduction using unsupervised learning is quite simple.
    Imagine a 3 x 3 image, with the following RGB color matrix:

    | Column1 | Column2 | Column3 |
    | ----- | ----- | ----- |
    | RGB(55, 180, 255) | RGB(230, 128, 7) | RGB(63, 12, 236) |
    | RGB(55, 180, 255) | RGB(230, 128, 7) | RGB(178, 172, 176) |
    | RGB(230, 128, 7) | RGB(98, 30, 56) | RGB(55, 180, 255) |

    From the table above, we can build a second matrix where each column
    corresponds to a color channel. So, for the previous table we could
    build a matrix with as follows:

    | Red | Green | Blue |
    | --- | --- | --- |
    | 55 | 180 | 255 |
    | 230 | 128 | 7 |
    | 63 | 12 | 236 |
    | $\\vdots$ | $\\vdots$ | $\\vdots$ |
    | 230 | 128 | 7 |
    | 98 | 30 | 56 |
    | 55 | 180 | 255 |

    Each of the columns in the the table above constitutes a variable,
    so that we can use a clustering algorithm (kmeans) to group the colors
    based on some discrimination metric, Euclidian distance, for example.
    Once we have each of the groups, we can map each of the colors of the 
    elements of each group to the predominant color of the group
    (the center of the group), and then restructure them to obtain the new image.
    """
