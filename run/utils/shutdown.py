import boto3
ids = ['i-0f88a515df7d24572']
ec2 = boto3.resource('ec2',
                     region_name='eu-west-2',
                     aws_access_key_id='AKIAIO5FODNN7EXAMPLE',
                     aws_secret_access_key='ABCDEF+c2L7yXeGvUyrPgYsDnWRRC1AYEXAMPLE'
                     )

ec2.instances.filter(InstanceIds = ids).stop() #for stopping an ec2 instance