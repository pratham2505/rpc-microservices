DESIGN.md

In this file I will explain the complete design behind my HW2 RPC microservices assignment.

I will explain everything I did in the section for which will tell:
1. Design
2. Tradeoffs and drawbacks of those decesions

Part 1: Basic RPC Implementation
1.1 Service Request Handling
1. def_handle_request()
    Design:
        What I did was decoded the incoming message, created a "Request" object, and then checked if the deadline passed and also applied optional fault injection, I also ran the calculation using "_calculate", and then set back either a success response or an error response. 

    Tradeoffs: 
    Benefits: It is really simple and easy to follow.  
               There is a clear seperation between request logic and the calculation logic.
               Works really well for small workloads.

    Drawbacks: Fault injection is basic and only for testing.
               It does not handle very high concurrency.
               Errors are all treated as internal errors without detailed categories.

2. HEALTH_CHECK Handling
    Design:
        I calculated the average latency from the recent request times, also packaged the services health info (healthy, current_load, average_latency) into a dictionary, and then sent it back as a HEALTH RESPONSE 

    Tradeoffs: 
    Benefits: It is a very simple implementation and easy to read and maintain.
               It only uses basic averages and no complex metrics.
    
    Drawbacks: Average latency may fluctuate heavily with using small samples.
               If many checks arrive rapidly, it still adds minor processing overhead.

1.2 _calculate()
    Design:
        I first converted the operation string to so the check is case insensitive:
        then checked operat = operation.lower 
        Then for each supported operation:
        1. sum - to get the total
        2. avg - to get the average
        3. min - to get the smallest value
        4. max - to get the largest value
        5. multiply - I start with the product = 1.0 and loop over the list and then multiply each value into product.
        if operat is non on then then I raise an error.

        Tradeoffs: 
        Benefits: The implementation is short, clear and  uses Python's optimized built in functions, which helps us reduce bugs and improves readability.

        Drawback: Build methods assume the list is non-empty, and for extremely large lists they may be less memory efficient than a straight approach.

Part 2: Load Balancing
2.1 - Round-Robin
    Design: 
        I always pick the next healthy instance in order using an index and back to the start when I reach the end of the list. I built a healthy_instances list that skips instances with open circuit breaks and if the rr_index is part the end of the list i reset it 0. I also use healthy instances[self.rr_index] as the selected instance. Then we use healthy_instances[self.rr_index] as the selected instance. And we update self.rr_index so the next call moves to the next server.

    Tradeoffs:
        Benefits: It is a simple logic with O(1) selection and predictable, and also even distribution of request across all the healthy servers.
    
        Drawbacks: This assumes that each request costs about the same. If one instance is much slower, it can still recieve the same number of request as the fast ones.

2.2 - Least Connections Strategy
    Design:
        I reuse the same healthy instances list and call min() to find the instance with the fewest active connections and then use that instance as selected_id.
    
    Tradeoffs: 
        Benefit: It automatically shifts more traffic to less loaded servers which also helps us balances work better.

        Drawbacks: The load balancer has to check every healthy instance each time so if there are many instances, this can get slower and take extra work.


2.3 - Statistics Tracking
    Design:
        I modified active connections using connections delta and made sure it never drops below zero and if a latency value exists I incresed total requests. If the request fails I also increased the error count. I updated the avg latency using the formula so I don't store al latencies.

    Tradeoffs:
        Benefit: It is efficient as I track stats without saving every latency value which keeps memory low and updates fast.

        Drawback: The running average can ge influenced by very old data since there is no reset of time window.

Part 3: Fault Tolerance (25 points) 
3.1 Retry Logic with Exponential Backoff 
    Design:
        I build a retry loop that which access the load balancer for a healthy instance and then check if the curcuit breaker doesn't exit. Before sending the request I update the stats and use breaker call to make the RPC call. On success update the stats and return the response, and on failure update the stats. Then I retry the delay with exponential backoff when failure happens
    
    Tradeoffs:
        Benefit: It greatly improves reliability by automatically recovering from temporary failures using smarter retry timing.

        Drawback: Exponential waits can become slow as attempts increase, and multiple retrying clients can add extra load to the system.

3.2 Circuit Breaker
    Design:
        I increased the total requests for every attempted call and if the status is OPEN first I check the current time - last failure time and if timeout has passed I move to HALF_OPEN and then reset success_Count and then update last_state_changed and raise an exception. If the state is HALF_OPEN or CLOSED I try to run func(*args, **kwargs) inside try/except. On success I call on_success and return the result. On error I call on_failure() and re raise the exception so the caller can retry or handle it.

    Tradeoffs:
        Benefit: Helps avoid repeatedly calling a failing service and gives it time to recover.

        Drawback: It adds extra checks on every call and sometimes the breaker might block calls even if the service recovered recently.

Part 4: Testing and Deployment 
4.1 Local Testing
    I already did the local testing and the output's screenshot is mentioned in the @DEPLOYMENT.md file.

4.2 Cloud Deployment
    I also did deploy on cloud distance to check with each service and the screenshots are mentioned in @DEPLOYMENT.md.


