import boto3
from botocore.client import Config


def main():
    ACCESS_KEY_ID = 'AKIAR67AE6TZCAB4R2BG'
    ACCESS_SECRET_KEY = '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'
    BUCKET_NAME = 'mlalgostoragecsv'

    data = open('trainableDataSigmoid_balanced_test.csv', 'rb')
    data2week = open('past_two_weekstrainDataSigmoid_balanced.csv', 'rb')

    s3 = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_SECRET_KEY,
        config=Config(signature_version='s3v4')
    )
    s3.Bucket(BUCKET_NAME).put_object(Key='trainableDataSigmoid_balanced_test.csv', Body=data, ACL='public-read')
    s3.Bucket(BUCKET_NAME).put_object(Key='past_two_weekstrainDataSigmoid_balanced.csv', Body=data2week, ACL='public-read')

    print ("Uploaded to AWS s3 Bucket mlalgostoragecsv")

if __name__ == '__main__':
    main()