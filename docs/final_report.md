# Weather-Based LIRR Delay Prediction Final Report

## Executive Summary
Long Island Rail Road delays can disrupt commuters’ schedules, especially when weather conditions make service less predictable. The goal of this project was to build a machine learning application that predicts how many minutes late an LIRR train is likely to be based on trip details and weather conditions.

To address this problem, our team combined historical LIRR delay records with Long Island weather data and trained regression models to estimate delay time in minutes. We tested a baseline Random Forest model and a Champion XGBoost model, and XGBoost consistently performed better across evaluation metrics. We then integrated the final model into a Streamlit application that allows users to select a departure station, arrival station, and date to receive a predicted delay along with a simple delay-status message. The final product provides commuters with a practical forecasting tool that can help them better anticipate possible delays and plan travel more effectively.

## The Data Journey
Our project used two primary datasets: historical LIRR delay data and Long Island weather data. The train dataset provided route information and observed delay times, while the weather dataset contributed factors such as precipitation, snowfall, snow depth, and temperature. These datasets were cleaned separately and then merged into a single modeling dataset for prediction.

One of the largest challenges was handling the number of categorical variables in the train data. The dataset we used contained around 49 departure stations, 43 arrival stations, 203 station pairs, and over 2,000 train identifiers. Representing all of these directly with one-hot encoding would have created a very high-dimensional dataset, so we needed a more efficient way to capture route and train patterns. Another challenge was aligning weather records with train trips so that each trip received the correct weather conditions for that day.

Outliers were also a major issue. Extremely large delays could distort the model and reduce its usefulness for day-to-day predictions, so we experimented with removing extreme values using a three-sigma rule. In one round of testing, this removed 4,559 rows. In a later version focused on 2022 and forward, 1,009 rows were removed. We also found evidence that the model may have been learning patterns from older years that did not generalize as well to recent service, which led us to test models using only newer data.

Feature engineering played a major role in the project. Important features included station pair, departure station mean delay, arrival station mean delay, train mean delay, and time-based variables such as year, month, day, and day of week. Weather variables included precipitation, snowfall, snow depth, maximum temperature, minimum temperature, and average temperature. In the application pipeline, additional engineered features such as cyclical time encodings and rolling weather or delay features were also used to support prediction.

To reduce the complexity of the categorical variables, we used mean encoding for train, departure station, arrival station, and station pair. This allowed the model to capture historical delay tendencies for these categories without creating an excessively sparse feature space.

## Modeling & Results
Because the target variable was numeric, minutes late, this project used regression models. Our baseline model was Random Forest, and our Champion model was XGBoost. XGBoost was ultimately selected because it consistently outperformed Random Forest across all major testing rounds.

We evaluated model performance using the following metrics:
-	Mean Absolute Error (MAE)
-	Root Mean Squared Error (RMSE)
-	R-squared
-	Mean Absolute Percentage Error (MAPE)

Initial testing on the broader dataset showed that XGBoost outperformed Random Forest:
-	Random Forest: MAE = 6.80, RMSE = 10.02, R² = -0.13, MAPE = 69.92%
-	XGBoost: MAE = 5.66, RMSE = 9.48, R² = -0.01, MAPE = 52.94%
After removing extreme outliers, both models improved:
-	Random Forest: MAE = 5.12, RMSE = 6.63, R² = -0.14, MAPE = 54.68%
-	XGBoost: MAE = 4.52, RMSE = 6.28, R² = -0.02, MAPE = 45.30%

We then adjusted the feature set by removing train_mean, which appeared to encourage overfitting. This led to additional improvement:
-	Random Forest: MAE = 4.88, RMSE = 6.44, R² = -0.08, MAPE = 51.05%
-	XGBoost: MAE = 4.41, RMSE = 6.14, R² = 0.02, MAPE = 43.94%
Our strongest modeling setup came from focusing on more recent data, using records from 2022 onward for training and evaluating on 2025 data:
-	Random Forest: MAE = 4.20, RMSE = 5.80, R² = -0.03, MAPE = 42.31%
-	XGBoost: MAE = 3.99, RMSE = 5.66, R² = 0.02, MAPE = 39.02%

Additional summary statistics from the saved XGBoost predictions showed approximately (results may vary vs personal XGB Metrics):
-	Actual mean delay: 11.11 minutes
-	Predicted mean delay: 10.35 minutes
-	MAE: 3.43 minutes
-	RMSE: 5.71 minutes
-   R²: 0.35
-	MAPE: 28.49 minutes

These results support selecting XGBoost as the Champion model. Across every major comparison, it produced lower error than Random Forest. Although the R² values remained low, indicating that delay prediction is still a noisy and difficult problem, the model still provided a reasonable estimate of likely delay length. In practice, reducing average absolute error to about 3.4 to 4.0 minutes suggests the model can offer useful rough guidance for commuters, even if it should not be treated as a perfect predictor.

## User Testing Impact
Sprint 5 user testing directly influenced the final design of the Streamlit application. Testers with different levels of computer experience interacted with the app and provided feedback on both usability and clarity.

Several users responded positively to the app’s simplicity. Melissa Anderson described the interface as easy to use once the stations and date were confirmed, and Josh Arbon was surprised that the task only required a few clicks. Tye Anderson also liked the typing and tab functionality in the station dropdowns, which made station selection faster and easier. These results confirmed that the basic input flow was simple and accessible.

At the same time, testing revealed several areas that needed improvement. Kate Shipley had difficulty understanding what the predicted delay minutes actually meant and felt that the weather information and prediction display were overwhelming. Cassidy Sterrett was confused by the filtered arrival station list and by the downloadable data shown at the bottom of the app. Melissa Anderson was briefly confused by the copy-link icon attached to the date input. Tye Anderson suggested that the results should be more visually prominent and that the weather information should appear higher on the page because it is central to the prediction logic.

This feedback directly shaped the final application. To make the output easier to interpret, the app includes a clear status indicator that translates predicted delay minutes into more user-friendly categories such as On Schedule, Moderate Delay, and Significant Delay. The app also maintains a streamlined input flow and limits arrival station options to historically valid station pairs, which reduces invalid selections. User testing also showed the importance of reducing clutter, simplifying how weather information is presented, and making prediction results easier to notice. Finally, the team recognized that a map may be helpful for some less experienced riders, but not essential for regular commuters, so route visualization was treated as optional rather than central to the main workflow.

## Recommendations & Future Work

Based on the model’s predictions, the main recommendation for stakeholders is to use weather-aware delay forecasting as a planning aid rather than as a replacement for official service alerts. For commuters, the model suggests that route history and weather conditions can meaningfully affect likely delays, so riders should check predicted delays before traveling and allow extra time during periods of precipitation, snowfall, or more severe weather. For LIRR or similar transit stakeholders, this type of tool could be incorporated into rider-facing systems to set expectations earlier and reduce uncertainty.

The results also suggest that recent historical data matters more than older records when building transit delay prediction systems. Stakeholders should prioritize up-to-date operational data when improving future forecasting tools so the model reflects current service patterns more accurately.

If we had another three months, there are several improvements we would add. First, we would connect the app to a real-time weather API so it could generate live predictions rather than relying on historical weather coverage. Second, we would continue tuning the XGBoost model and test additional modeling approaches to improve predictive power. Third, we would further refine the application by simplifying the weather display, improving result visibility, and adding clearer explanations for why some arrival stations are filtered. We would also consider optional route visualization and commuter guidance features for less experienced riders. Finally, incorporating richer operational data, such as service disruptions, maintenance activity, and peak versus off-peak context, could improve model accuracy and make the system more useful in practice.

## Conclusion
This project demonstrated that machine learning can be used to estimate LIRR delays from weather and route-related information in a way that is meaningful for end users. By combining transportation and weather data, engineering route-based features, comparing regression models, and responding to user testing feedback, our team developed a practical prediction tool for commuters. XGBoost emerged as the strongest model, and the final application translated its predictions into a more accessible interface that supports better travel planning.