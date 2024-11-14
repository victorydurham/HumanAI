import json
import pandas as pd
import numpy as np
""" Note: This program does not include action data. Action data synchronisation 
    is completed seperately and manually by inspecting screen recordings."""
def load_json():
    # Load json predictions file from Hume AI Video Model
    with open('Non_Academic_Data.json', 'r') as file:
        return json.load(file)

def process_data(json_data):
    # Create lists to store data
    timestamps = []
    thought_data = []
    # Create dictionaries for face, prosody, and language data. 
    # The selection of emotions can be chosen as you wish but must correspond to existing emotions in the json file
    emotions_face = {emotion: [] for emotion in [
        "Interest (Face)", "Boredom (Face)", "Joy (Face)", "Anger (Face)",
        "Surprise (positive) (Face)", "Disappointment (Face)", "Satisfaction (Face)", "Confusion (Face)"
    ]}
    emotions_prosody = {emotion: [] for emotion in [
        "Interest (Prosody)", "Boredom (Prosody)", "Joy (Prosody)", "Anger (Prosody)",
        "Surprise (positive) (Prosody)", "Disappointment (Prosody)", "Satisfaction (Prosody)", "Confusion (Prosody)"
    ]}
    emotions_language = {emotion: [] for emotion in [
        "Interest (Language)", "Boredom (Language)", "Joy (Language)", "Anger (Language)",
        "Surprise (positive) (Language)", "Disappointment (Language)", "Satisfaction (Language)", "Confusion (Language)"
    ]}

    # Extract data from JSON
    for entry in json_data:
        if 'results' not in entry or 'predictions' not in entry['results']:
            print("'results' or 'predictions' key not found in entry.")
            continue  # Skip this entry

        for prediction in entry['results']['predictions']:
            if 'models' not in prediction:
                print("'models' key not found in prediction.")
                continue  # Skip this prediction

            models = prediction['models']

            # Extract face predictions
            face_predictions = []
            if 'face' in models and models['face']['grouped_predictions']:
                face_grouped_predictions = models['face']['grouped_predictions']
                # Collect all face predictions
                for group in face_grouped_predictions:
                    face_predictions.extend(group['predictions'])
                print(f"Number of face predictions: {len(face_predictions)}")
            else:
                print("No 'face' predictions available.")

            # Build a dictionary of face data indexed by time
            face_data_by_time = {pred['time']: pred for pred in face_predictions if 'time' in pred}
            face_times = sorted(face_data_by_time.keys())

            # Extract prosody predictions
            prosody_predictions = []
            if 'prosody' in models and models['prosody']['grouped_predictions']:
                prosody_grouped_predictions = models['prosody']['grouped_predictions']
                # Collect all prosody predictions
                for group in prosody_grouped_predictions:
                    prosody_predictions.extend(group['predictions'])
                print(f"Number of prosody predictions: {len(prosody_predictions)}")
            else:
                print("No 'prosody' predictions available.")
                continue  # Can't proceed without prosody predictions

            # Extract language predictions
            language_predictions = []
            if 'language' in models and models['language']['grouped_predictions']:
                language_grouped_predictions = models['language']['grouped_predictions']
                # Collect all language predictions
                for group in language_grouped_predictions:
                    language_predictions.extend(group['predictions'])
                print(f"Number of language predictions: {len(language_predictions)}")
            else:
                print("No 'language' predictions available.")
                continue  # Can't proceed without language predictions

            # Collect unique emotion names from language predictions
            unique_language_emotions = set()
            for pred in language_predictions:
                if 'emotions' in pred:
                    for emotion in pred['emotions']:
                        unique_language_emotions.add(emotion['name'])
            print("Unique emotion names in language predictions:")
            print(unique_language_emotions)

            # Build a list of language predictions with their time intervals
            language_predictions_with_time = []
            for pred in language_predictions:
                if 'time' in pred:
                    lang_begin = pred['time']['begin']
                    lang_end = pred['time']['end']
                    language_predictions_with_time.append((lang_begin, lang_end, pred))

            # Iterate over prosody predictions
            for prosody in prosody_predictions:
                if 'time' not in prosody or 'emotions' not in prosody:
                    continue  # Skip invalid entries
                prosody_begin = prosody['time']['begin']
                prosody_end = prosody['time']['end']
                prosody_mid = (prosody_begin + prosody_end) / 2
                timestamps.append(prosody_mid)
                thought_data.append(prosody.get('text', ''))

                # Find face predictions within this prosody time interval
                matching_face_times = [t for t in face_times if prosody_begin <= t <= prosody_end]

                # Find language predictions that overlap with this prosody prediction
                matching_language_predictions = [
                    pred for (lang_begin, lang_end, pred) in language_predictions_with_time
                    if lang_begin <= prosody_end and lang_end >= prosody_begin
                ]

                # Initialize emotion scores
                face_emotion_scores = {emotion: np.nan for emotion in emotions_face}
                prosody_emotion_scores = {emotion: 0 for emotion in emotions_prosody}
                language_emotion_scores = {emotion: np.nan for emotion in emotions_language}

                # Aggregate face emotions
                if matching_face_times:
                    for emotion in emotions_face.keys():
                        emotion_key = emotion.replace(' (Face)', '')  # Remove ' (Face)' from the end
                        scores = []
                        for t in matching_face_times:
                            face = face_data_by_time[t]
                            face_emotion_score = next(
                                (item['score'] for item in face['emotions'] if item['name'] == emotion_key), np.nan
                            )
                            scores.append(face_emotion_score)
                        # Average the scores, ignoring NaNs
                        face_emotion_scores[emotion] = np.nanmean(scores)
                else:
                    # No matching face times, set emotions to NaN
                    face_emotion_scores = {emotion: np.nan for emotion in emotions_face}

                # Extract prosody emotions
                for emotion in emotions_prosody.keys():
                    emotion_key = emotion.replace(' (Prosody)', '')  # Remove ' (Prosody)' from the end
                    prosody_emotion_score = next(
                        (item['score'] for item in prosody['emotions'] if item['name'] == emotion_key), 0
                    )
                    prosody_emotion_scores[emotion] = prosody_emotion_score

                # Aggregate language emotions
                if matching_language_predictions:
                    for emotion in emotions_language.keys():
                        emotion_key = emotion.replace(' (Language)', '')  # Remove ' (Language)' from the end
                        scores = []
                        for language_pred in matching_language_predictions:
                            # Check if 'emotions' key exists
                            if 'emotions' in language_pred:
                                for item in language_pred['emotions']:
                                    if item['name'].lower() == emotion_key.lower():
                                        language_emotion_score = item.get('score', np.nan)
                                        if language_emotion_score is not None:
                                            scores.append(language_emotion_score)
                                        else:
                                            print(f"No score found for emotion '{item['name']}' in language prediction.")
                            else:
                                print(f"'emotions' key missing in language prediction: {language_pred}")
                        # Average the scores, ignoring NaNs
                        if scores:
                            language_emotion_scores[emotion] = np.nanmean(scores)
                        else:
                            language_emotion_scores[emotion] = np.nan
                else:
                    # No matching language predictions, set emotions to NaN
                    language_emotion_scores = {emotion: np.nan for emotion in emotions_language}

                # Append emotion scores
                for emotion in emotions_face:
                    emotions_face[emotion].append(face_emotion_scores[emotion])
                for emotion in emotions_prosody:
                    emotions_prosody[emotion].append(prosody_emotion_scores[emotion])
                for emotion in emotions_language:
                    emotions_language[emotion].append(language_emotion_scores[emotion])


    data = {'Timestamp': timestamps, 'Thought Data': thought_data}
    data.update(emotions_face)
    data.update(emotions_prosody)
    data.update(emotions_language)
    df = pd.DataFrame(data)
    return df

def main():
    json_data = load_json()
    df = process_data(json_data)
    print(df.head())

  
    df.to_excel('Synchronised2.xlsx', index=False)

if __name__ == "__main__":
    main()
