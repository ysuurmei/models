import boto3
ids = ['i-0f88a515df7d24572']
ec2 = boto3.resource('ec2')
ec2.instances.filter(InstanceIds = ids).stop() #for stopping an ec2 instance