from google.cloud import storage


def create_bucket_class_location(bucket_name, credential_path):
    """
    Create a new bucket in the US region with the coldline storage
    class
    """
    storage_client = storage.Client.from_service_account_json(credential_path)

    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "STANDARD"
    new_bucket = storage_client.create_bucket(bucket, location="us")

    print(
        "Created bucket {} in {} with storage class {}".format(
            new_bucket.name, new_bucket.location, new_bucket.storage_class
        )
    )
    return new_bucket


def upload_file(bucket_name, credential_path, source_file_name, blob_name):
    """
    Upload file to google cloud
    """
    storage_client = storage.Client.from_service_account_json(credential_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_file_name)


# create_bucket_class_location("ml2-project2", "ml2-gonzalo-romero-b7cef8301abc.json")
upload_file(
    "ml2-project", "ml2-gonzalo-romero-b7cef8301abc.json", "./test.csv", "test.csv"
)
upload_file(
    "ml2-project", "ml2-gonzalo-romero-b7cef8301abc.json", "./train.csv", "train.csv"
)
