# ============================================================
# TOURISM EXPERIENCE ANALYTICS
# Streamlit Application
# ============================================================

# ------------------------------------------------------------
# STEP 1: IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

# Import operating-system utilities
import os

# Import joblib for loading saved ML models and recommendation artifacts
import joblib

# Import NumPy for numerical operations
import numpy as np

# Import Pandas for data manipulation
import pandas as pd

# Import Streamlit for building the web application
import streamlit as st

# Import Matplotlib for visualization
import matplotlib.pyplot as plt

# Import Seaborn for attractive statistical charts
import seaborn as sns


# ------------------------------------------------------------
# STEP 2: STREAMLIT PAGE CONFIGURATION
# ------------------------------------------------------------

# Configure the Streamlit page
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)


# ------------------------------------------------------------
# STEP 3: DEFINE PATHS
# ------------------------------------------------------------

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the outputs folder
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ------------------------------------------------------------
# STEP 4: LOAD SAVED DATA AND MODELS
# ------------------------------------------------------------

# Create a cached function so models are not reloaded on every interaction
@st.cache_resource
def load_models():

    # Load the regression model
    regression_model = joblib.load(
        os.path.join(OUTPUT_DIR, "regression_model.pkl")
    )

    # Load the classification model
    classification_model = joblib.load(
        os.path.join(OUTPUT_DIR, "classification_model.pkl")
    )

    # Load the label encoders
    encoders = joblib.load(
        os.path.join(OUTPUT_DIR, "label_encoders.pkl")
    )

    # Load the user-item matrix
    user_item_matrix = joblib.load(
        os.path.join(OUTPUT_DIR, "user_item_matrix.pkl")
    )

    # Load item-item similarity matrix
    item_similarity_df = joblib.load(
        os.path.join(OUTPUT_DIR, "item_similarity.pkl")
    )

    # Load content similarity matrix
    content_similarity_df = joblib.load(
        os.path.join(OUTPUT_DIR, "content_similarity.pkl")
    )

    # Load attraction-level features
    attraction_features = joblib.load(
        os.path.join(OUTPUT_DIR, "attraction_features.pkl")
    )

    # Return all loaded artifacts
    return (
        regression_model,
        classification_model,
        encoders,
        user_item_matrix,
        item_similarity_df,
        content_similarity_df,
        attraction_features
    )


# ------------------------------------------------------------
# STEP 5: LOAD CLEANED MASTER DATASET
# ------------------------------------------------------------

# Create a cached function for loading the cleaned dataset
@st.cache_data
def load_data():

    # Define the cleaned master dataset path
    data_path = os.path.join(
        OUTPUT_DIR,
        "cleaned_tourism_master.csv"
    )

    # Check whether the file exists
    if not os.path.exists(data_path):
        return None

    # Load the cleaned dataset
    df = pd.read_csv(data_path)

    # Return the dataframe
    return df


# ------------------------------------------------------------
# STEP 6: LOAD ALL ARTIFACTS
# ------------------------------------------------------------

try:

    # Load all saved ML and recommendation artifacts
    (
        regression_model,
        classification_model,
        encoders,
        user_item_matrix,
        item_similarity_df,
        content_similarity_df,
        attraction_features
    ) = load_models()

    # Load the cleaned master dataset
    master = load_data()

except Exception as e:

    # Display an error message if artifacts cannot be loaded
    st.error("Unable to load project artifacts.")

    # Display the actual error for debugging
    st.exception(e)

    # Stop application execution
    st.stop()


# ------------------------------------------------------------
# STEP 7: APPLICATION TITLE
# ------------------------------------------------------------

# Display main application title
st.title("🌍 Tourism Experience Analytics")

# Display subtitle
st.markdown(
    """
    **Classification • Rating Prediction • Personalized Recommendation**
    
    Analyze tourism behavior, predict visitor preferences and ratings,
    and generate personalized attraction recommendations.
    """
)


# ------------------------------------------------------------
# STEP 8: SIDEBAR NAVIGATION
# ------------------------------------------------------------

# Display sidebar title
st.sidebar.title("Navigation")

# Create navigation menu
page = st.sidebar.radio(
    "Select Module",
    [
        "📊 Dashboard",
        "🤖 Prediction",
        "🎯 Recommendation System"
    ]
)


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    # Display dashboard heading
    st.header("📊 Tourism Analytics Dashboard")

    # Check whether cleaned data is available
    if master is None:

        # Display warning
        st.warning(
            "cleaned_tourism_master.csv was not found in the outputs folder."
        )

    else:

        # ----------------------------------------------------
        # KPI SECTION
        # ----------------------------------------------------

        # Calculate number of transactions
        total_transactions = len(master)

        # Calculate number of users
        total_users = master["UserId"].nunique()

        # Calculate number of attractions
        total_attractions = master["Attraction"].nunique()

        # Calculate average rating
        average_rating = master["Rating"].mean()

        # Create four KPI columns
        col1, col2, col3, col4 = st.columns(4)

        # Display transaction count
        col1.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

        # Display user count
        col2.metric(
            "Total Users",
            f"{total_users:,}"
        )

        # Display attraction count
        col3.metric(
            "Attractions",
            f"{total_attractions:,}"
        )

        # Display average rating
        col4.metric(
            "Average Rating",
            f"{average_rating:.2f} / 5"
        )

        # Add spacing
        st.markdown("---")

        # ----------------------------------------------------
        # RATING DISTRIBUTION
        # ----------------------------------------------------

        # Create two columns
        col1, col2 = st.columns(2)

        with col1:

            # Display chart heading
            st.subheader("⭐ Rating Distribution")

            # Create rating count table
            rating_counts = (
                master["Rating"]
                .value_counts()
                .sort_index()
            )

            # Create Matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 5))

            # Create bar chart
            sns.barplot(
                x=rating_counts.index,
                y=rating_counts.values,
                ax=ax
            )

            # Set chart title
            ax.set_title("Distribution of Visitor Ratings")

            # Set X-axis label
            ax.set_xlabel("Rating")

            # Set Y-axis label
            ax.set_ylabel("Number of Ratings")

            # Display chart
            st.pyplot(fig)

            # Close figure
            plt.close(fig)

        with col2:

            # Display chart heading
            st.subheader("👥 Visit Mode Distribution")

            # Create visit-mode count table
            mode_counts = (
                master["VisitModeName"]
                .value_counts()
                .sort_values(ascending=False)
            )

            # Create Matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 5))

            # Create bar chart
            sns.barplot(
                x=mode_counts.values,
                y=mode_counts.index,
                ax=ax
            )

            # Set chart title
            ax.set_title("Visitor Mode Distribution")

            # Set X-axis label
            ax.set_xlabel("Number of Visitors")

            # Set Y-axis label
            ax.set_ylabel("Visit Mode")

            # Display chart
            st.pyplot(fig)

            # Close figure
            plt.close(fig)

        # ----------------------------------------------------
        # TOP ATTRACTIONS
        # ----------------------------------------------------

        st.subheader("🏆 Top Attractions")

        # Count transactions by attraction
        top_attractions = (
            master.groupby("Attraction")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create horizontal bar chart
        top_attractions.plot(
            kind="barh",
            ax=ax
        )

        # Set title
        ax.set_title("Top 10 Most Visited Attractions")

        # Set X-axis label
        ax.set_xlabel("Number of Visits")

        # Set Y-axis label
        ax.set_ylabel("Attraction")

        # Display chart
        st.pyplot(fig)

        # Close figure
        plt.close(fig)

        # ----------------------------------------------------
        # YEARLY RATING TREND
        # ----------------------------------------------------

        st.subheader("📈 Average Rating by Year")

        # Calculate yearly average rating
        yearly_rating = (
            master.groupby("VisitYear")["Rating"]
            .mean()
            .reset_index()
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))

        # Create line chart
        sns.lineplot(
            data=yearly_rating,
            x="VisitYear",
            y="Rating",
            marker="o",
            ax=ax
        )

        # Set title
        ax.set_title("Average Tourism Rating by Year")

        # Set X-axis label
        ax.set_xlabel("Visit Year")

        # Set Y-axis label
        ax.set_ylabel("Average Rating")

        # Display chart
        st.pyplot(fig)

        # Close figure
        plt.close(fig)


# ============================================================
# PAGE 2: PREDICTION
# ============================================================

elif page == "🤖 Prediction":

    # Display page heading
    st.header("🤖 Tourism Prediction")

    # Create prediction tabs
    prediction_tab1, prediction_tab2 = st.tabs(
        [
            "👥 Visit Mode Classification",
            "⭐ Rating Prediction"
        ]
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    with prediction_tab1:

        # Display section heading
        st.subheader("Predict Visitor Visit Mode")

        # Explanation
        st.write(
            "Enter the visitor and attraction information to predict "
            "the most likely visit mode."
        )

        # Create input columns
        col1, col2 = st.columns(2)

        with col1:

            # Select user
            user_id = st.number_input(
                "User ID",
                min_value=1,
                value=int(user_item_matrix.index[0])
            )

            # Select attraction
            attraction_options = sorted(
                attraction_features["Attraction"].astype(str).unique()
            )

            selected_attraction = st.selectbox(
                "Attraction",
                attraction_options
            )

            # Select visit year
            visit_year = st.number_input(
                "Visit Year",
                min_value=2000,
                max_value=2100,
                value=2022
            )

            # Select visit month
            visit_month = st.slider(
                "Visit Month",
                min_value=1,
                max_value=12,
                value=6
            )

        with col2:

            # Rating is used by the classification model
            rating = st.slider(
                "Rating",
                min_value=1,
                max_value=5,
                value=4
            )

            # Find attraction information
            attraction_row = attraction_features[
                attraction_features["Attraction"]
                == selected_attraction
            ]

            # Stop if attraction cannot be found
            if attraction_row.empty:

                # Display error
                st.error("Attraction information not found.")

                # Stop this section
                st.stop()

            # Extract attraction type
            attraction_type = attraction_row.iloc[0][
                "AttractionTypeId"
            ]

            # Extract average rating
            attraction_avg_rating = attraction_row.iloc[0][
                "AvgRating"
            ]

            # Extract popularity
            attraction_popularity = attraction_row.iloc[0][
                "Popularity"
            ]

            # Select geographic information
            continent_id = st.number_input(
                "Continent ID",
                min_value=0,
                value=1
            )

            # Select region
            region_id = st.number_input(
                "Region ID",
                min_value=0,
                value=1
            )

            # Select country
            country_id = st.number_input(
                "Country ID",
                min_value=0,
                value=1
            )

            # Select city
            city_id = st.number_input(
                "City ID",
                min_value=-1,
                value=1
            )

        # Add space
        st.markdown("---")

        # Create prediction button
        predict_mode_button = st.button(
            "🔮 Predict Visit Mode",
            type="primary"
        )

        # Execute prediction
        if predict_mode_button:

            try:

                # ------------------------------------------------
                # ENCODE CATEGORICAL FEATURES
                # ------------------------------------------------

                # Create helper function for encoding
                def encode_feature(
                    encoder_name,
                    value
                ):

                    # Get encoder
                    encoder = encoders[encoder_name]

                    # Convert value to string
                    value = str(value)

                    # Check whether value exists in encoder classes
                    if value not in encoder.classes_:

                        # Use first known encoded value
                        return int(
                            encoder.transform(
                                [encoder.classes_[0]]
                            )[0]
                        )

                    # Return encoded value
                    return int(
                        encoder.transform([value])[0]
                    )

                # Encode continent
                continent_encoded = encode_feature(
                    "ContinentId",
                    continent_id
                )

                # Encode region
                region_encoded = encode_feature(
                    "RegionId",
                    region_id
                )

                # Encode country
                country_encoded = encode_feature(
                    "CountryId",
                    country_id
                )

                # Encode attraction type
                attraction_type_encoded = encode_feature(
                    "AttractionTypeId",
                    attraction_type
                )

                # ------------------------------------------------
                # BUILD CLASSIFICATION INPUT
                # ------------------------------------------------

                # Create input dataframe
                classification_input = pd.DataFrame(
                    {
                        "VisitYear": [visit_year],
                        "VisitMonth": [visit_month],
                        "Rating": [rating],
                        "ContinentId": [continent_encoded],
                        "RegionId": [region_encoded],
                        "CountryId": [country_encoded],
                        "CityId": [city_id],
                        "AttractionTypeId": [
                            attraction_type_encoded
                        ],
                        "AttractionAvgRating": [
                            attraction_avg_rating
                        ],
                        "AttractionPopularity": [
                            attraction_popularity
                        ]
                    }
                )

                # Make classification prediction
                prediction = classification_model.predict(
                    classification_input
                )[0]

                # Convert prediction to original visit mode
                predicted_mode = encoders[
                    "VisitModeName"
                ].inverse_transform(
                    [prediction]
                )[0]

                # Display prediction
                st.success(
                    f"Predicted Visit Mode: **{predicted_mode}**"
                )

                # Check whether probability prediction is available
                if hasattr(
                    classification_model,
                    "predict_proba"
                ):

                    # Calculate class probabilities
                    probabilities = (
                        classification_model.predict_proba(
                            classification_input
                        )[0]
                    )

                    # Get class names
                    class_names = encoders[
                        "VisitModeName"
                    ].classes_

                    # Create probability dataframe
                    probability_df = pd.DataFrame(
                        {
                            "Visit Mode": class_names,
                            "Probability": probabilities
                        }
                    )

                    # Convert to percentage
                    probability_df[
                        "Probability"
                    ] *= 100

                    # Sort probabilities
                    probability_df = (
                        probability_df
                        .sort_values(
                            "Probability",
                            ascending=False
                        )
                    )

                    # Display probabilities
                    st.subheader(
                        "Prediction Probabilities"
                    )

                    # Display dataframe
                    st.dataframe(
                        probability_df,
                        use_container_width=True
                    )

            except Exception as e:

                # Display prediction error
                st.error(
                    "Classification prediction failed."
                )

                # Display technical error
                st.exception(e)

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    with prediction_tab2:

        # Display heading
        st.subheader("Predict Attraction Rating")

        # Explanation
        st.write(
            "Enter visitor and attraction information to predict "
            "the expected attraction rating."
        )

        # Create two columns
        col1, col2 = st.columns(2)

        with col1:

            # User ID
            regression_user_id = st.number_input(
                "User ID",
                min_value=1,
                value=int(user_item_matrix.index[0]),
                key="reg_user"
            )

            # Attraction
            regression_attraction = st.selectbox(
                "Attraction",
                attraction_options,
                key="reg_attraction"
            )

            # Visit year
            regression_year = st.number_input(
                "Visit Year",
                min_value=2000,
                max_value=2100,
                value=2022,
                key="reg_year"
            )

            # Visit month
            regression_month = st.slider(
                "Visit Month",
                min_value=1,
                max_value=12,
                value=6,
                key="reg_month"
            )

        with col2:

            # Visit mode selection
            visit_modes = [
                "Business",
                "Couples",
                "Family",
                "Friends",
                "Solo"
            ]

            selected_mode = st.selectbox(
                "Visit Mode",
                visit_modes
            )

            # Geographic IDs
            regression_continent = st.number_input(
                "Continent ID",
                min_value=0,
                value=1,
                key="reg_continent"
            )

            regression_region = st.number_input(
                "Region ID",
                min_value=0,
                value=1,
                key="reg_region"
            )

            regression_country = st.number_input(
                "Country ID",
                min_value=0,
                value=1,
                key="reg_country"
            )

            regression_city = st.number_input(
                "City ID",
                min_value=-1,
                value=1,
                key="reg_city"
            )

        # Prediction button
        predict_rating_button = st.button(
            "⭐ Predict Rating",
            type="primary"
        )

        # Execute prediction
        if predict_rating_button:

            try:

                # Find selected attraction
                attraction_row = attraction_features[
                    attraction_features["Attraction"]
                    == regression_attraction
                ]

                # Stop if attraction is unavailable
                if attraction_row.empty:

                    # Display error
                    st.error(
                        "Attraction information not found."
                    )

                else:

                    # Extract attraction information
                    attraction_type = attraction_row.iloc[0][
                        "AttractionTypeId"
                    ]

                    # Extract attraction average rating
                    attraction_avg_rating = attraction_row.iloc[0][
                        "AvgRating"
                    ]

                    # Extract popularity
                    attraction_popularity = attraction_row.iloc[0][
                        "Popularity"
                    ]

                    # Encode categorical features
                    continent_encoded = encode_feature(
                        "ContinentId",
                        regression_continent
                    )

                    region_encoded = encode_feature(
                        "RegionId",
                        regression_region
                    )

                    country_encoded = encode_feature(
                        "CountryId",
                        regression_country
                    )

                    attraction_type_encoded = encode_feature(
                        "AttractionTypeId",
                        attraction_type
                    )

                    # ------------------------------------------------
                    # HISTORICAL USER AVERAGE RATING
                    # ------------------------------------------------

                    # Default to overall average
                    global_rating = master[
                        "Rating"
                    ].mean()

                    # Check whether user exists
                    if (
                        master is not None
                        and regression_user_id
                        in master["UserId"].values
                    ):

                        # Calculate historical user average
                        user_avg_rating = master.loc[
                            master["UserId"]
                            == regression_user_id,
                            "Rating"
                        ].mean()

                    else:

                        # Use global average for unseen users
                        user_avg_rating = global_rating

                    # ------------------------------------------------
                    # BUILD REGRESSION INPUT
                    # ------------------------------------------------

                    regression_input = pd.DataFrame(
                        {
                            "VisitYear": [
                                regression_year
                            ],
                            "VisitMonth": [
                                regression_month
                            ],
                            "VisitMode": [
                                encoders[
                                    "VisitModeName"
                                ].transform(
                                    [selected_mode]
                                )[0]
                            ],
                            "ContinentId": [
                                continent_encoded
                            ],
                            "RegionId": [
                                region_encoded
                            ],
                            "CountryId": [
                                country_encoded
                            ],
                            "CityId": [
                                regression_city
                            ],
                            "AttractionTypeId": [
                                attraction_type_encoded
                            ],
                            "UserAvgRating": [
                                user_avg_rating
                            ],
                            "AttractionAvgRating": [
                                attraction_avg_rating
                            ],
                            "AttractionPopularity": [
                                attraction_popularity
                            ]
                        }
                    )

                    # Make rating prediction
                    predicted_rating = regression_model.predict(
                        regression_input
                    )[0]

                    # Keep prediction within valid rating range
                    predicted_rating = np.clip(
                        predicted_rating,
                        1,
                        5
                    )

                    # Display result
                    st.success(
                        f"Predicted Rating: "
                        f"**{predicted_rating:.2f} / 5**"
                    )

            except Exception as e:

                # Display error
                st.error(
                    "Rating prediction failed."
                )

                # Display technical details
                st.exception(e)


# ============================================================
# PAGE 3: RECOMMENDATION SYSTEM
# ============================================================

elif page == "🎯 Recommendation System":

    # Display heading
    st.header("🎯 Personalized Attraction Recommendation")

    # Display explanation
    st.write(
        """
        The recommendation system combines **item-based collaborative
        filtering** and **content-based filtering** into a hybrid
        recommendation system.
        """
    )

    # --------------------------------------------------------
    # USER SELECTION
    # --------------------------------------------------------

    # Convert user IDs to a list
    user_ids = list(user_item_matrix.index)

    # Create user selection
    selected_user = st.selectbox(
        "Select User ID",
        user_ids
    )

    # Select number of recommendations
    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    # --------------------------------------------------------
    # COLLABORATIVE RECOMMENDATION FUNCTION
    # --------------------------------------------------------

    def recommend_collaborative(
        user_id,
        top_n=5
    ):

        # Check whether user exists
        if user_id not in user_item_matrix.index:

            # Return empty series
            return pd.Series(dtype=float)

        # Get ratings for selected user
        user_ratings = user_item_matrix.loc[user_id]

        # Keep attractions already rated by user
        rated = user_ratings[
            user_ratings > 0
        ]

        # Check whether user has ratings
        if rated.empty:

            # Return empty recommendation
            return pd.Series(dtype=float)

        # Calculate weighted similarity scores
        scores = (
            item_similarity_df[rated.index]
            .dot(rated)
            / rated.sum()
        )

        # Remove already-rated attractions
        scores = scores.drop(
            labels=rated.index,
            errors="ignore"
        )

        # Return top recommendations
        return (
            scores
            .sort_values(
                ascending=False
            )
            .head(top_n)
        )


    # --------------------------------------------------------
    # CONTENT-BASED RECOMMENDATION FUNCTION
    # --------------------------------------------------------

    def recommend_content_based(
        attraction_name,
        top_n=5
    ):

        # Check whether attraction exists
        if (
            attraction_name
            not in content_similarity_df.index
        ):

            # Return empty series
            return pd.Series(dtype=float)

        # Get similarity scores
        scores = (
            content_similarity_df[
                attraction_name
            ]
            .drop(
                attraction_name
            )
            .sort_values(
                ascending=False
            )
        )

        # Return top recommendations
        return scores.head(top_n)


    # --------------------------------------------------------
    # HYBRID RECOMMENDATION FUNCTION
    # --------------------------------------------------------

    def recommend_hybrid(
        user_id,
        top_n=5
    ):

        # Generate collaborative recommendations
        collaborative = recommend_collaborative(
            user_id,
            top_n
        )

        # Get user history
        user_history = master[
            master["UserId"] == user_id
        ].sort_values(
            "Rating",
            ascending=False
        )

        # If user has no history, return collaborative results
        if user_history.empty:

            return collaborative

        # Select highest-rated attraction
        top_liked = user_history.iloc[0][
            "Attraction"
        ]

        # Generate content recommendations
        content = recommend_content_based(
            top_liked,
            top_n
        )

        # Combine collaborative and content scores
        combined = pd.concat(
            [
                collaborative,
                content
            ]
        )

        # Average scores for duplicate attractions
        combined = (
            combined
            .groupby(level=0)
            .mean()
            .sort_values(
                ascending=False
            )
        )

        # Get attractions already visited
        already_visited = (
            user_history["Attraction"]
            .unique()
        )

        # Remove attractions already visited
        combined = combined.drop(
            labels=already_visited,
            errors="ignore"
        )

        # Return top recommendations
        return combined.head(top_n)


    # --------------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # --------------------------------------------------------

    if st.button(
        "✨ Generate Recommendations",
        type="primary"
    ):

        try:

            # Generate hybrid recommendations
            recommendations = recommend_hybrid(
                selected_user,
                top_n
            )

            # Check whether recommendations exist
            if recommendations.empty:

                # Display warning
                st.warning(
                    "No recommendations available "
                    "for this user."
                )

            else:

                # Display heading
                st.subheader(
                    "✨ Suggested Attractions"
                )

                # Create recommendation dataframe
                recommendation_df = (
                    recommendations
                    .reset_index()
                )

                # Rename columns
                recommendation_df.columns = [
                    "Attraction",
                    "Recommendation Score"
                ]

                # Add ranking column
                recommendation_df.insert(
                    0,
                    "Rank",
                    range(
                        1,
                        len(
                            recommendation_df
                        ) + 1
                    )
                )

                # Display recommendations
                st.dataframe(
                    recommendation_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ------------------------------------------------
                # USER HISTORY
                # ------------------------------------------------

                st.subheader(
                    "📜 User Rating History"
                )

                # Get selected user's history
                user_history = master[
                    master["UserId"]
                    == selected_user
                ][
                    [
                        "Attraction",
                        "Rating",
                        "VisitYear",
                        "VisitMonth",
                        "VisitModeName"
                    ]
                ].sort_values(
                    "Rating",
                    ascending=False
                )

                # Display history
                st.dataframe(
                    user_history,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            # Display recommendation error
            st.error(
                "Recommendation generation failed."
            )

            # Display technical details
            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

# Add horizontal separator
st.markdown("---")

# Display footer
st.caption(
    "Tourism Experience Analytics | "
    "Classification • Regression • Recommendation"
)