import sys
import paramiko

hostname="flogo4.local"
username="poppy"
password="poppy"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=hostname, username=username, password=password)
stdin, stdout, stderr = ssh.exec_command("aplay ~/beep1.wav")
print(stdout.read())
ssh.close()