README.md

In this file I will share how to run the code:

First to test it locally first you run the code locally by using this command "python rpc_assignment.py test". Now to run the code on cloud what you do is, first you create 3 instances which are shared in the screenshot in DEPLOYMENT.md with the following instructions: e2-micro, Spot, Ubuntu 20.04, us-central1-a. You also have create a firewall rule for all the 3 ports by following these instructions: Name: allow-rpc-services, Protocols and ports: TCP 9000-9002 Source IPv4 ranges: 0.0.0.0/0. After doing this you click SSH and launch the terminal in browser for each instance and you upload the code manually first and then run this command: 
1. nohup python3 rpc_assignment.py server 900X > service.log 2>&1 &
What this does: It starts the python server in background and keeps it running if I close the terminal and saves all its output and errors in service.log

This way you can do for all the rpc service instances you created for different ports and run the code on cloud by swapping the localhost's IP Address to the EXTERNAL IP from the 3 vm instances created.