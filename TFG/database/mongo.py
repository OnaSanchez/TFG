from pymongo import MongoClient
import pandas as pd


############################# FUNCTIONS FOR DATA TRANSFER WITH MONGODB ##################################

def connect_db(database_name, host, port):
    """
    Input:

    database_name (str): The name of the MongoDB database to connect to.
    host (str): The MongoDB server's host address.
    port (int): The port number on which MongoDB is running.

    Description: Establishes a connection to the MongoDB server.

    Output: The MongoDB database object (db) and the MongoClient object (connection).

    """

    dsn = "mongodb://{}:{}".format(host, port)
    connection = MongoClient(dsn)

    # Select the database to use
    return (connection[database_name], connection)



def users_collection(db, df_pacients, df_professionals):
    """
    Input:

    db (MongoDB database object): The MongoDB database object.
    df_pacients (DataFrame): DataFrame containing patient data.
    df_professionals (DataFrame): DataFrame containing professional data.

    Description: Creates a new collection named "Usuari" in the MongoDB database and inserts patient and professional data into the collection.

    Output: The created collection.

    """

    # If the collection already exists, drop it
    if "Usuari" in db.list_collection_names():
        collection = db["Usuari"]
        collection.drop()

    # Create a new collection named "Usuari"
    collection = db.create_collection("Usuari")

    # Convert DataFrame to a list of dictionaries for easier insertion
    pacients_data = df_pacients.to_dict(orient='records')
    professionals_data = df_professionals.to_dict(orient='records')

    # Insert patient data into the collection
    collection.insert_many(pacients_data)

    # Insert professional data into the collection
    collection.insert_many(professionals_data)

    # Success message
    print("Patient and professional data added successfully to the 'Usuari' collection.")

    return collection



def session_collection(db, collection, df_sessio):
    """
    Input:

    db (MongoDB database object): The MongoDB database object.
    collection (MongoDB collection object): The "Usuari" collection.
    df_sessio (DataFrame): DataFrame containing session data.

    Description: Creates a new "sessio" collection in the MongoDB database, adding the corresponding patiend ID, and inserts session data into the collection.

    Output: The created collection and the session data.
    """

    # If the 'sessio' collection already exists, drop it
    if "sessio" in db.list_collection_names():
        collection_sessio = db["sessio"]
        collection_sessio.drop()

    # Create a new 'sessio' collection
    collection_sessio = db.create_collection("sessio")

    # Get a list of IDs and user names from the 'Usuari' collection
    usuarios_data = list(collection.find({}, {'_id': 1, 'ID': 1, 'Nom_Usuari': 1}))

    # Create a dictionary to map user names to IDs
    nom_id_mapping = {usuario['Nom_Usuari']: usuario['_id'] for usuario in usuarios_data}
    df_sessio['ID'] = df_sessio['Nom_Usuari'].map(nom_id_mapping)

    # Remove the 'Nom_Usuari' column if it is no longer needed
    df_sessio = df_sessio.drop(columns=['Nom_Usuari'])

    # Convert the 'Data' and 'Hora' columns to the desired format
    df_sessio['Data'] = pd.to_datetime(df_sessio['Data'], format='%Y-%m-%d', errors='coerce')
    df_sessio['Hora'] = df_sessio['Hora'].astype(str)  # Convertir a cadena

    # Convert DataFrame to a list of dictionaries for easier insertion
    sessio_data = df_sessio.to_dict(orient='records')
    # Insert session data into the 'sessio' collection
    collection_sessio.insert_many(sessio_data)

    # Success message
    print("Session data added successfully to the 'sessio' collection.")

    return collection_sessio, sessio_data



def test_collection(db, df_test, sessio_data):
    """
    Input:

    db (MongoDB database object): The MongoDB database object.
    df_test (DataFrame): DataFrame containing psychometric test data.
    sessio_data (list): List of dictionaries containing session data.

    Description: Creates a new "test_psico" collection in the MongoDB database and inserts psychometric test data into the collection, linking the data to sessions.

    Output: The created collection.
    """

    # If the 'test_psico' collection already exists, drop it
    if "test_psico" in db.list_collection_names():
        collection_psico = db["test_psico"]
        collection_psico.drop()

    # Create a new 'test_psico' collection
    collection_psico = db.create_collection("test_psico")

    # Convert DataFrame to a list of dictionaries for easier insertion
    test_data = df_test.to_dict(orient="records")

    # Insert data into the MongoDB collection
    for entry in test_data:
        # Extract session and estado from the entry
        state = entry.pop("stage")

        # Insert each test with attributes session, estado, test_name, and result
        for test_name, result in entry.items():
            if (test_name != "ID"):
                sessio = entry['ID']
                for pyros in sessio_data:
                    if pyros['ID_sessio'] == sessio:
                        id_user = pyros['ID']
                        id_session = pyros['ID_sessio']

                test_entry = {
                    "Estat": state,
                    "Nom_test": test_name,
                    "Puntuacio_total_test": result,
                    "ID": id_user,
                    "ID_sessio": id_session
                }

                # Insert the test entry into collection_psico
                collection_psico.insert_one(test_entry)

    # Success message
    print("Psychometric tests data added successfully to the 'test_psico' collection.")

    return collection_psico



def signal_collection(db, measures, names):
    """
    Input:

    db (MongoDB database object): The MongoDB database object.
    measures (list of DataFrames): List of DataFrames containing signal measures.
    names (list): List of signal names corresponding to the measures.

    Description: Creates a new "Senyal" collection in the MongoDB database and inserts signal data into the collection.

    Output: The created collection.
    """

    # If the 'Senyal' collection already exists, drop it
    if "Senyal" in db.list_collection_names():
        collection_senyal = db["Senyal"]
        collection_senyal.drop()

    # Create a new 'Senyal' collection
    collection_senyal = db.create_collection("Senyal")

    for df_signal, signal_name in zip(measures, names):
        # Initialize the current state
        current_stage = None

        # Iterate through the rows of the DataFrame
        for index, row in df_signal.iterrows():

            # Get the state of the current row --> Assuming there will be 3 signals per session (separated by stage)
            stage = row["stage"]
            session = row['ID']
            # Check for a change in state

            if stage != current_stage:
                # Create the entry for the 'Senyal' collection
                signal_entry = {
                    "Nom_senyal": signal_name,
                    "Estat": stage,
                    "ID_sessio": session
                    # "Senyal": extract signal information,
                }

                # Insert the entry into the 'Senyal' collection
                collection_senyal.insert_one(signal_entry)

                # Update the current state
                current_stage = stage

    # Success message
    print("Signal data added successfully to the 'Senyal' collection.")

    return collection_senyal



def measure_collection(db, measures, names):
    """
    Input:

    db (MongoDB database object): The MongoDB database object.
    measures (list of DataFrames): List of DataFrames containing signal measures.
    names (list): List of signal names corresponding to the measures.

    Description: Creates a new "Mesura" collection in the MongoDB database and inserts measurement data into the collection, linking the data to signals, sessions, and users.

    Output: The created collection.

    """

    # If the 'Mesura' collection already exists, drop it
    if "Mesura" in db.list_collection_names():
        collection_mesura = db["Mesura"]
        collection_mesura.drop()

    # Create a new 'Mesura' collection
    collection_mesura = db.create_collection("Mesura")

    # Columns that are not considered as measures
    cols_no_measures = ['ID', 'stage', 'onset', 'offset', 'length', 'ECGname']

    for df_signal, signal_name in zip(measures, names):

        # Obtain the attributes of each measurement
        for index, row in df_signal.iterrows():

            id_session = row['ID']
            stage = row['stage']

            # 'onset' is only applicable for non-saliv signals
            if signal_name != 'saliv':
                onset = row['onset']
            else:
                onset = None

            # Query the 'sessio' collection to obtain the corresponding 'ID'
            user_id_query = db.sessio.find_one({'ID_sessio': id_session})
            if user_id_query:
                user_id = user_id_query['ID']
            else:
                user_id = None  # Handle the case where no corresponding 'ID' is found

            # Query the 'Senyal' collection to obtain the corresponding '_id'
            senyal_query = db.Senyal.find_one({'ID_sessio': id_session, 'Estat': stage, 'Nom_senyal': signal_name})
            if senyal_query:
                id_senyal = senyal_query['_id']
            else:
                id_senyal = None  # Handle the case where no corresponding '_id' is found

            # Iterate through columns of the DataFrame
            for column in df_signal.columns:
                if row[column] == 0:
                    value = None
                else:
                    value = row[column]
                if column not in cols_no_measures:
                    measure_doc = {
                        'ID_senyal': id_senyal,
                        'Nom_var': column,
                        'Estat': stage,
                        'ID_sessio': id_session,
                        'ID': user_id,
                        'Valor': value,
                        'Minut': onset
                    }

                    # Insert the document into the 'Mesura' collection in MongoDB
                    db.Mesura.insert_one(measure_doc)

    # Success message
    print("Measures data added successfully to the 'Mesura' collection.")

    return collection_mesura