# Movie Recommender System

This project is a movie recommendation system that uses content-based filtering to recommend movies based on their genres, keywords, cast, and crew. It leverages the TMDB 5000 Movies dataset and TMDB 5000 Credits dataset to create movie recommendations.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Merges TMDB movie and credits datasets
- Cleans and preprocesses data by handling missing values and converting data formats
- Uses natural language processing techniques like stemming
- Converts text data into numerical features using CountVectorizer
- Computes cosine similarity between movies
- Recommends movies based on content similarity
- Provides a web interface built with Streamlit to interact with the recommendation system

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/movie-recommender-system.git
    cd movie-recommender-system
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the datasets and place them in the project directory:
    - `tmdb_5000_movies.csv`
    - `tmdb_5000_credits.csv`

5. (Optional) Download NLTK resources:
    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

1. Preprocess the data and train the model:
    ```bash
    python preprocess_and_train.py
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3. Open the provided local URL in your web browser to access the app.

## Data Preprocessing

The preprocessing script:
- Loads the TMDB movies and credits datasets
- Merges the datasets on the `title` column
- Selects relevant columns: `movie_id`, `title`, `genres`, `keywords`, `overview`, `cast`, `crew`
- Handles missing values and duplicates
- Converts `genres`, `keywords`, `cast`, and `crew` columns from JSON-like strings to lists
- Applies stemming to the `tags` column created by concatenating `overview`, `genres`, `keywords`, `cast`, and `crew`
- Uses CountVectorizer to convert the `tags` column into numerical features
- Calculates cosine similarity between the feature vectors
- Saves the processed data and similarity matrix using pickle

## Model Training

The model training process:
- Uses CountVectorizer to convert text data into numerical data
- Calculates cosine similarity between movies based on the feature vectors
- Saves the processed data and similarity matrix for later use

## Streamlit App

The Streamlit app:
- Loads the processed data and similarity matrix
- Provides a user interface to input a movie name and get recommendations
- Displays recommended movies with their posters and ratings

## Future Improvements

- Add collaborative filtering to enhance recommendations
- Include more features like release date and runtime
- Improve the user interface with more filters and sorting options

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- **Name**: Shaik Yousuf
- **LinkedIn**: [Shaik Yousuf](https://www.linkedin.com/in/shaik-yousuf-a39566228/)

Feel free to contact me if you have any questions or suggestions!
