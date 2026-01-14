#!/usr/bin/env python3
"""
Microservices with RPC and Load Balancing
"""

import socket
import json
import time
import threading
import random
import struct
import hashlib
import queue
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


# ============================================================
# CONFIGURATION: Edit this section for cloud deployment
# ============================================================

# For local testing (default):
# SERVICE_INSTANCES = [
#     ('localhost', 9000),
#     ('localhost', 9001),
#     ('localhost', 9002),
# ]

SERVICE_INSTANCES = [
    ('35.188.116.66', 9000),
    ('35.192.125.99', 9001),
    ('34.41.39.201', 9002),
]

# For cloud deployment, uncomment and replace with your GCP instance IPs:
# SERVICE_INSTANCES = [
#     ('35.232.123.45', 9000),  # Replace with your instance-0 IP
#     ('35.232.123.46', 9001),  # Replace with your instance-1 IP  
#     ('35.232.123.47', 9002),  # Replace with your instance-2 IP
# ]

# ============================================================
# End of configuration section
# ============================================================

# ===================== PROTOCOL DEFINITION =====================
# DO NOT MODIFY THIS SECTION

class MessageType(Enum):
    """RPC message types"""
    REQUEST = 1
    RESPONSE = 2
    HEALTH_CHECK = 3
    HEALTH_RESPONSE = 4

class StatusCode(Enum):
    """Response status codes"""
    OK = 0
    CANCELLED = 1
    DEADLINE_EXCEEDED = 2
    NOT_FOUND = 3
    UNAVAILABLE = 4
    INTERNAL_ERROR = 5

@dataclass
class Request:
    """RPC Request structure"""
    request_id: str
    method: str
    operation: str
    values: List[float]
    deadline: float  # Unix timestamp
    metadata: Dict[str, str]

@dataclass
class Response:
    """RPC Response structure"""
    request_id: str
    status: StatusCode
    result: Optional[float]
    error_message: Optional[str]
    latency_ms: float
    server_id: str

class Protocol:
    """Wire protocol for RPC communication - PROVIDED"""
    
    @staticmethod
    def encode_message(msg_type: MessageType, data: dict) -> bytes:
        """Encode message for transmission"""
        json_data = json.dumps(data).encode('utf-8')
        length = len(json_data) + 1
        header = struct.pack('!IB', length, msg_type.value)
        return header + json_data
    
    @staticmethod
    def decode_message(sock: socket.socket) -> Tuple[Optional[MessageType], Optional[dict]]:
        """Decode received message"""
        try:
            header = sock.recv(5)
            if not header or len(header) < 5:
                return None, None
            length, msg_type = struct.unpack('!IB', header)
            data = b''
            while len(data) < length - 1:
                chunk = sock.recv(min(4096, length - 1 - len(data)))
                if not chunk:
                    return None, None
                data += chunk
            return MessageType(msg_type), json.loads(data.decode('utf-8'))
        except Exception as e:
            print(f"Protocol decode error: {e}")
            return None, None

# ===================== SERVICE IMPLEMENTATION =====================

class ServiceInstance:
    """Microservice instance - STUDENT MUST IMPLEMENT MARKED SECTIONS"""
    
    def __init__(self, instance_id: str, port: int):
        self.instance_id = instance_id
        self.port = port
        self.socket = None
        self.running = False
        
        # Service state
        self.current_load = 0
        self.max_load = 100
        self.healthy = True
        self.processing_times = deque(maxlen=100)
        
        # Fault injection (for testing)
        self.inject_latency = 0
        self.error_rate = 0.0
    
    def start(self):
        """Start service instance"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('', self.port))
        self.socket.listen(10)
        self.running = True
        
        print(f"Service {self.instance_id} listening on port {self.port}")
        
        while self.running:
            try:
                client_sock, addr = self.socket.accept()
                if self.current_load < self.max_load:
                    threading.Thread(target=self.handle_request, 
                                   args=(client_sock,), daemon=True).start()
                else:
                    self._send_error(client_sock, StatusCode.UNAVAILABLE, 
                                   "Service overloaded")
                    client_sock.close()
            except:
                break
    
    def handle_request(self, client_sock: socket.socket):
        """Handle incoming RPC request"""
        self.current_load += 1
        start_time = time.time()
        
        try:
            msg_type, payload = Protocol.decode_message(client_sock)
            
            if msg_type == MessageType.REQUEST:
                # TODO: STUDENT MUST IMPLEMENT
                # 1. Parse request from data dictionary: request = Request(**data)
                # 2. Check if deadline has passed: if time.time() > request.deadline
                # 3. Apply fault injection if configured (for testing):
                #    - if self.inject_latency > 0: time.sleep(self.inject_latency / 1000.0)
                #    - if random.random() < self.error_rate: raise error
                # 4. Perform the calculation using self._calculate()
                # 5. Create Response object with appropriate status and result
                # 6. Send response using Protocol.encode_message()
                
                # Your code here...
                # We first start with parsing request from data dictionary
                request = Request(
                    request_id=payload.get('request_id'),
                    method=payload.get('method'),
                    operation=payload.get('operation'),
                    values=payload.get('values', []),
                    deadline=payload.get('deadline', 0.0),
                    metadata=payload.get('metadata', {})
                )
    
                # Then we check if deadline has passed 
                current_timestamp = time.time()
                if current_timestamp > request.deadline:
                    # if it has passed we send deadline exceeded response
                    deadline_err = {
                        'request_id': request.request_id,
                        'status': StatusCode.DEADLINE_EXCEEDED.value,
                        'result': None,
                        'error_message': 'Request expired before processing',
                        'latency_ms': (current_timestamp - start_time) * 1000,
                        'server_id': self.instance_id
                    }
                    client_sock.send(Protocol.encode_message(MessageType.RESPONSE, deadline_err))
                    return

                # if self.inject_latency > 0 we simulate latency
                if self.inject_latency > 0:
                    time.sleep(self.inject_latency / 1000.0)

                is_failed = (random.random() < self.error_rate)
                
                # if random.random() < self.error_rate we raise error
                if is_failed:
                    fault_resp = {
                        'request_id': request.request_id,
                        'status': StatusCode.INTERNAL_ERROR.value,
                        'result': None,
                        'error_message': 'Simulated internal failure',
                        'latency_ms': (time.time() - start_time) * 1000,
                        'server_id': self.instance_id
                    }
                    client_sock.send(Protocol.encode_message(MessageType.RESPONSE, fault_resp))
                    return

                # Then we perform the calculation using self._calculate()
                try:
                    calc_output = self._calculate(request.operation, request.values)
                    
                    success_resp = {
                        'request_id': request.request_id,
                        'status': StatusCode.OK.value,
                        'result': calc_output,
                        'error_message': None,
                        'latency_ms': (time.time() - start_time) * 1000,
                        'server_id': self.instance_id
                    }
                    client_sock.send(Protocol.encode_message(MessageType.RESPONSE, success_resp))
                
                except ValueError as logic_err:
                    logic_err_resp = {
                        'request_id': request.request_id,
                        'status': StatusCode.INTERNAL_ERROR.value,
                        'result': None,
                        'error_message': str(logic_err),
                        'latency_ms': (time.time() - start_time) * 1000,
                        'server_id': self.instance_id
                    }
                    client_sock.send(Protocol.encode_message(MessageType.RESPONSE, logic_err_resp))
                
                return

            elif msg_type == MessageType.HEALTH_CHECK:
                # TODO: STUDENT MUST IMPLEMENT
                # Return health status including:
                # - healthy: self.healthy
                # - current_load: self.current_load
                # - average_latency: calculate from self.processing_times
                
                # Your code here...
                avg_lat = 0.0
    
                # Calculate average latency from processing_times    
                if self.processing_times:
                    total_time = sum(self.processing_times)
                    count = len(self.processing_times)
                    avg_lat = (total_time / count) * 1000.0

                # Then we build health data dictionary
                health_data = {
                    'healthy': self.healthy,
                    'current_load': self.current_load,
                    'average_latency': avg_lat
                }
                # and we send health response
                client_sock.send(Protocol.encode_message(MessageType.HEALTH_RESPONSE, health_data))
                return
                
        except Exception as unexpected_err:
            print(f"Server exception: {unexpected_err}")
            self._send_error(client_sock, StatusCode.INTERNAL_ERROR, str(unexpected_err))
            
        finally:
            self.current_load -= 1
            duration = time.time() - start_time
            self.processing_times.append(duration)
            client_sock.close()          
    
    def _calculate(self, operation: str, values: List[float]) -> float:
        """Perform calculation based on operation"""
        # TODO: STUDENT MUST IMPLEMENT
        # Support operations: sum, avg, min, max, multiply
        # Raise ValueError for unknown operations
        
        # Your code here...
        operat = operation.lower()

        if operat == "sum":
            return float(sum(values))
        elif operat == "avg":
            return float(sum(values) / len(values))
        elif operat == "min":
            return float(min(values))
        elif operat == "max":
            return float(max(values))
        elif operat == "multiply":
            product = 1.0
            for i in range (len(values)):
                product *= values[i]
            return float(product)

        else:
            raise ValueError(f"unknown operation: {operation}")
    
    def _send_error(self, sock: socket.socket, status: StatusCode, message: str):
        """Send error response - PROVIDED"""
        response = {
            'request_id': 'error',
            'status': status.value,
            'result': None,
            'error_message': message,
            'latency_ms': 0,
            'server_id': self.instance_id
        }
        sock.send(Protocol.encode_message(MessageType.RESPONSE, response))
    
    def shutdown(self):
        """Shutdown service"""
        self.running = False
        if self.socket:
            self.socket.close()

# ===================== LOAD BALANCING =====================

class LoadBalancingStrategy(Enum):
    """Available load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"

@dataclass
class InstanceInfo:
    """Service instance information"""
    instance_id: str
    host: str
    port: int
    weight: int = 1
    active_connections: int = 0
    total_requests: int = 0
    error_count: int = 0
    avg_latency: float = 0.0
    last_health_check: float = 0
    healthy: bool = True
    circuit_breaker_state: str = "closed"

class LoadBalancer:
    """Load balancer - STUDENT MUST IMPLEMENT STRATEGIES"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.instances: Dict[str, InstanceInfo] = {}
        self.instance_list: List[str] = []
        
        # Round-robin state
        self.rr_index = 0
        
        # Statistics
        self.request_count = 0
        self.distribution = defaultdict(int)
        
        # Thread safety
        self.lock = threading.Lock()
    
    def add_instance(self, instance: InstanceInfo):
        """Add service instance to pool"""
        with self.lock:
            self.instances[instance.instance_id] = instance
            self.instance_list.append(instance.instance_id)
    
    def remove_instance(self, instance_id: str):
        """Remove service instance from pool"""
        with self.lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                self.instance_list.remove(instance_id)
    
    def select_instance(self, request: Request) -> Optional[InstanceInfo]:
        """Select instance for request based on strategy"""
        with self.lock:
            # Filter healthy instances
            healthy_instances = [
                iid for iid in self.instance_list 
                if self.instances[iid].healthy and 
                   self.instances[iid].circuit_breaker_state != "open"
            ]
            
            if not healthy_instances:
                return None
            
            selected_id = None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                # TODO: STUDENT MUST IMPLEMENT
                # Round-robin: Select next instance in order
                # Use self.rr_index and update it (with wraparound)
                
                # Your code here...
                if self.rr_index >= len(healthy_instances):
                    self.rr_index = 0

                # select the instance id at the current rr_index
                selected_id = healthy_instances[self.rr_index]

                self.rr_index = (self.rr_index + 1) % len(healthy_instances)

            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                # TODO: STUDENT MUST IMPLEMENT
                # Select instance with least active connections
                # Use instance.active_connections to decide
                
                # Your code here...
                if healthy_instances:
                    selected_id = min(
                        healthy_instances,
                        key=lambda iid: self.instances[iid].active_connections
                    )
                
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                # TODO: STUDENT MUST IMPLEMENT
                # Random selection from healthy instances
                
                # Your code here...
                if healthy_instances:
                    selected_id = random.choice(healthy_instances)

            # Update statistics
            if selected_id:
                self.distribution[selected_id] += 1
                self.request_count += 1
                return self.instances[selected_id]
            
            return None
    
    def update_instance_stats(self, instance_id: str, latency: float, 
                            success: bool, connections_delta: int = 0):
        """Update instance statistics after request"""
        # TODO: STUDENT MUST IMPLEMENT
        # Update the instance's statistics:
        # - active_connections (add connections_delta)
        # - total_requests (increment if this was a request)
        # - error_count (increment if not success)
        # - avg_latency (running average: (old_avg * old_count + new_latency) / new_count)
        
        with self.lock:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                # Your code here...
                #First we update the active connections and add the connections delta to ensure it does not go below 0
                instance.active_connections += connections_delta
                if instance.active_connections < 0:
                    instance.active_connections = 0  

                # Then we update total requests 
                if latency is not None:
                    instance.total_requests += 1

                    # Then we update error count if the request was not successful
                    if not success: 
                        instance.error_count += 1

                    # Then we update average latency using running average formula
                    old_avg = instance.avg_latency
                    old_count = instance.total_requests - 1  

                    if old_count <= 0:
                        instance.avg_latency = latency
                    else:
                        instance.avg_latency = (old_avg * old_count + latency) / instance.total_requests


    
    def get_stats(self) -> dict:
        """Get load balancer statistics"""
        with self.lock:
            stats = {
                'total_requests': self.request_count,
                'strategy': self.strategy.value,
                'instances': {}
            }
            
            for iid, info in self.instances.items():
                stats['instances'][iid] = {
                    'requests': self.distribution[iid],
                    'percentage': (self.distribution[iid] / self.request_count * 100) 
                                if self.request_count > 0 else 0,
                    'healthy': info.healthy,
                    'avg_latency': info.avg_latency,
                    'active_connections': info.active_connections,
                    'errors': info.error_count
                }
            
            return stats

# ===================== CIRCUIT BREAKER =====================

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests  
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for fault tolerance - STUDENT MUST IMPLEMENT"""
    
    def __init__(self, instance_id: str,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 success_threshold: int = 3):
        self.instance_id = instance_id
        self.state = CircuitBreakerState.CLOSED
        
        # Configuration
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        
        # Metrics
        self.total_requests = 0
        self.rejected_requests = 0
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        # TODO: STUDENT MUST IMPLEMENT
        # 1. Check current state
        # 2. If OPEN:
        #    - Check if recovery_timeout has passed
        #    - If yes, transition to HALF_OPEN
        #    - If no, reject request (raise exception)
        # 3. If HALF_OPEN or CLOSED:
        #    - Try to execute function
        #    - Call on_success() if successful
        #    - Call on_failure() if failed
        # 4. Return result or raise exception
        
        self.total_requests += 1
        
        # Your code here...
        current = time.time()

        # Check current state
        if self.state == CircuitBreakerState.OPEN:
            # If the state is open we check if the recovery timeout has passed
            if (current - self.last_failure_time) >= self.recovery_timeout:
                # if yes we transition to HALF_OPEN
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.last_state_change = current
            else:
                # or if its no we reject the request and raise exception 
                self.rejected_requests += 1
                raise Exception(f"Circuit open for instance {self.instance_id}")

        # If the state is HALF_OPEN or CLOSED we execute the function call on if on_success or on_failure if failed
        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception:
            self.on_failure()
            raise

    
    def on_success(self):
        """Handle successful call"""
        # TODO: STUDENT MUST IMPLEMENT
        # Update state based on success:
        # - If CLOSED: reset failure count
        # - If HALF_OPEN: increment success count
        #   - If success_count >= success_threshold: transition to CLOSED
        
        # Your code here...
        current = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            # Normal operation – clear any previous failures
            self.failure_count = 0

        elif self.state == CircuitBreakerState.HALF_OPEN:
            # We are testing if the instance has recovered
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                # Enough successful calls – close the circuit
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.last_state_change = current

    
    def on_failure(self):
        """Handle failed call"""
        # TODO: STUDENT MUST IMPLEMENT
        # Update state based on failure:
        # - If CLOSED: increment failure count
        #   - If failure_count >= failure_threshold: transition to OPEN
        # - If HALF_OPEN: transition to OPEN immediately
        # - Record last_failure_time = time.time()
        
        # Your code here...
        current = time.time()
        self.last_failure_time = current

        # if failure occurs we increment failure count
        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1

            # if failuer count >= failuer threshold we open the circuit
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.last_state_change = current

        # If the state is HALF_OPEN then we transition to OPEN immediately we also record the last failure time
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.failure_count = self.failure_threshold  
            self.last_state_change = current 

    
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        return self.state.value

# ===================== SMART CLIENT =====================

class RetryStrategy(Enum):
    """Retry strategies"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"

class SmartClient:
    """Client with retry logic and circuit breakers - STUDENT MUST IMPLEMENT"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Configuration
        self.max_retries = 3
        self.retry_strategy = RetryStrategy.EXPONENTIAL
        self.base_delay = 100  # milliseconds
        self.timeout = 5.0  # seconds
        
        # Metrics
        self.request_log = []
    
    def send_request(self, request: Request) -> Response:
        """Send request with retries and circuit breaking"""
        # TODO: STUDENT MUST IMPLEMENT
        # 1. Try up to max_retries times:
        #    a. Select instance using load_balancer
        #    b. Get/create circuit breaker for instance
        #    c. Use circuit breaker to call _send_single_request
        #    d. Update load_balancer stats with connections_delta=+1 before, -1 after
        #    e. If successful, return response
        #    f. If failed, calculate retry delay and wait
        # 2. If all retries exhausted, raise exception
        
        attempt = 0
        last_error = None
        
        # Your code here...
        while attempt < self.max_retries:
            attempt += 1

            # First we select instance using load balancer
            instance = self.load_balancer.select_instance(request)
            if instance is None:
                last_error = Exception("No healthy instances available")
                break

            instance_id = instance.instance_id

            # Then we create circuit breaker to call _send_single_request
            if instance_id not in self.circuit_breakers:
                self.circuit_breakers[instance_id] = CircuitBreaker(instance_id)

            breaker = self.circuit_breakers[instance_id]

            # Then we update load balancer stats to active_connections +1 (before request)
            self.load_balancer.update_instance_stats(instance_id, latency=None, success=True, connections_delta=+1)

            try:
                # Then we use circuit breaker to call _send_single_request
                response = breaker.call(self._send_single_request, instance, request)

                # Then we update load_balancer stats to active_connections -1 (after)
                self.load_balancer.update_instance_stats(instance_id, latency=response.latency_ms, success=True, connections_delta=-1)

                # If it is successful then we return response
                return response

            except Exception as e:
                last_error = e

                # Update load_balancer stats to active_connections -1 (after failure)
                self.load_balancer.update_instance_stats(instance_id, latency=0, success=False, connections_delta=-1)

                # If it is failed we calculate retry delay and wait
                if attempt < self.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    time.sleep(delay / 1000.0)

        # If all retries exhausted raise we raise exception
        raise Exception(f"Request failed after {self.max_retries} attempts: {last_error}")

    
    def _send_single_request(self, instance: InstanceInfo, request: Request) -> Response:
        """Send single request to instance"""
        # TODO: STUDENT MUST IMPLEMENT
        # 1. Create socket and set timeout (self.timeout)
        # 2. Connect to instance (host, port)
        # 3. Send request using Protocol.encode_message
        # 4. Receive response using Protocol.decode_message
        # 5. Parse response and return Response object
        # 6. Handle socket errors appropriately
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        
        try:
            # Your code here...
            # Connect to service instance
            sock.connect((instance.host, instance.port))

            # Build request dict
            dict = {
                "request_id": request.request_id,
                "method": request.method,
                "operation": request.operation,
                "values": request.values,
                "deadline": request.deadline,
                "metadata": request.metadata,
            }

            # Sending the encoded REQUEST
            sock.sendall(
                Protocol.encode_message(MessageType.REQUEST, dict)
            )

            # Then we Receive response using Protocol.decode_message
            msg_type, data = Protocol.decode_message(sock)
            if msg_type != MessageType.RESPONSE or data is None:
                raise Exception("Invalid response from server")

            # The we parse response dict into Response object
            status = StatusCode(data["status"])
            response = Response(
                request_id=data["request_id"],
                status=status,
                result=data.get("result"),
                error_message=data.get("error_message"),
                latency_ms=data.get("latency_ms", 0.0),
                server_id=data.get("server_id", instance.instance_id),
            )

            # If server reports not OK we treat as failure so we can retry
            if response.status != StatusCode.OK:
                raise Exception(
                    f"Server error ({response.status.name}): {response.error_message}"
                )

            # Then we Log the request
            self.request_log.append(
                {
                    "instance_id": instance.instance_id,
                    "request_id": request.request_id,
                    "status": response.status.name,
                    "latency_ms": response.latency_ms,
                }
            )
            # Then we retrn the response
            return response
        finally:
            sock.close()
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay in milliseconds"""
        # TODO: STUDENT MUST IMPLEMENT
        # Based on self.retry_strategy:
        # - EXPONENTIAL: base_delay * (2 ** attempt)
        # - LINEAR: base_delay * attempt
        # - FIXED: base_delay
        
        # Your code here...
        if self.retry_strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        elif self.retry_strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        else:
            delay = self.base_delay

        return float(delay)


# ===================== TESTING FRAMEWORK =====================

class Tester:
    """Testing framework - PROVIDED"""
    
    def test_basic_functionality(self, client: SmartClient):
        """Test basic RPC functionality"""
        print("\n=== Testing Basic Functionality ===")
        
        test_cases = [
            ("sum", [1, 2, 3, 4, 5], 15),
            ("avg", [10, 20, 30], 20),
            ("min", [5, 2, 8, 1], 1),
            ("max", [5, 2, 8, 1], 8),
            ("multiply", [2, 3, 4], 24),
        ]
        
        passed = 0
        for operation, values, expected in test_cases:
            try:
                request = Request(
                    request_id=f"test_{operation}_{time.time()}",
                    method="Calculate",
                    operation=operation,
                    values=values,
                    deadline=time.time() + 5,
                    metadata={}
                )
                
                response = client.send_request(request)
                
                if response.status == StatusCode.OK and response.result == expected:
                    print(f"✓ {operation} test passed")
                    passed += 1
                else:
                    print(f"✗ {operation} test failed: got {response.result}, expected {expected}")
            except Exception as e:
                print(f"✗ {operation} test failed with error: {e}")
        
        print(f"Passed {passed}/{len(test_cases)} tests")
        return passed == len(test_cases)
    
    def test_load_balancing(self, client: SmartClient, num_requests: int = 100):
        """Test load distribution with sequential and concurrent requests"""
        print(f"\n=== Testing Load Balancing ({num_requests} requests) ===")
        
        # Part 1: Sequential requests (shows basic distribution)
        print("Part 1: Sequential requests")
        for i in range(num_requests):
            request = Request(
                request_id=f"lb_test_{i}",
                method="Calculate",
                operation="sum",
                values=[1, 2, 3],
                deadline=time.time() + 5,
                metadata={}
            )
            try:
                client.send_request(request)
            except:
                pass
        
        stats = client.load_balancer.get_stats()
        print(f"Strategy: {stats['strategy']}")
        print("Distribution:")
        for instance_id, instance_stats in stats['instances'].items():
            print(f"  {instance_id}: {instance_stats['requests']} requests "
                  f"({instance_stats['percentage']:.1f}%)")
        
        # Part 2: Concurrent requests (shows least-connections behavior)
        if client.load_balancer.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            print("\nPart 2: Concurrent requests (tests least-connections properly)")
            results_lock = threading.Lock()
            results = []
            
            def send_concurrent_request(i):
                request = Request(
                    request_id=f"concurrent_test_{i}",
                    method="Calculate",
                    operation="sum",
                    values=[1, 2, 3],
                    deadline=time.time() + 10,
                    metadata={}
                )
                try:
                    response = client.send_request(request)
                    with results_lock:
                        results.append(response)
                except Exception as e:
                    pass
            
            # Send 50 requests concurrently
            threads = []
            for i in range(50):
                thread = threading.Thread(target=send_concurrent_request, args=(i,))
                thread.start()
                threads.append(thread)
                
                # Small delay to create overlap but not wait for completion
                if i % 10 == 0:
                    time.sleep(0.005)
            
            # Wait for all to complete
            for thread in threads:
                thread.join()
            
            print(f"Completed {len(results)} concurrent requests")
            print("Note: With concurrent load, least-connections should adapt to varying instance speeds")
    
    def test_fault_tolerance(self, instances: List[ServiceInstance], client: SmartClient):
        """Test resilience to failures"""
        print("\n=== Testing Fault Tolerance ===")
        
        # Inject failures
        if len(instances) > 0:
            instances[0].error_rate = 0.5
            print(f"Injected 50% error rate into {instances[0].instance_id}")
        
        # Send requests and measure success rate
        success = 0
        total = 50
        
        for i in range(total):
            request = Request(
                request_id=f"fault_test_{i}",
                method="Calculate",
                operation="sum",
                values=[1, 2],
                deadline=time.time() + 5,
                metadata={}
            )
            try:
                response = client.send_request(request)
                if response.status == StatusCode.OK:
                    success += 1
            except:
                pass
        
        success_rate = (success / total) * 100
        print(f"Success rate with failures: {success_rate:.1f}%")
        
        # Test circuit breaker by causing many failures
        if len(instances) > 0:
            instances[0].error_rate = 1.0
            print(f"\nInjected 100% error rate into {instances[0].instance_id}")
        
        # Circuit should open and reject quickly
        start = time.time()
        for i in range(10):
            try:
                request = Request(
                    request_id=f"cb_test_{i}",
                    method="Calculate",
                    operation="sum",
                    values=[1],
                    deadline=time.time() + 5,
                    metadata={}
                )
                client.send_request(request)
            except:
                pass
        
        elapsed = time.time() - start
        print(f"Time for 10 requests with circuit breaker: {elapsed:.2f}s")
        print("Note: Circuit breaker should open after threshold failures")
        
        # Reset error rate
        if len(instances) > 0:
            instances[0].error_rate = 0.0

# ===================== MAIN EXECUTION =====================

def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 rpc_assignment.py [server|demo|test]")
        print("  server <port> - Start a service instance")
        print("  demo          - Run basic demonstration")
        print("  test          - Run comprehensive test suite")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "server":
        # Start a single service instance
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9000
        instance = ServiceInstance(f"instance_{port}", port)
        try:
            instance.start()
        except KeyboardInterrupt:
            print(f"\nShutting down instance on port {port}")
            instance.shutdown()
    
    elif mode == "demo":
        # Run a demonstration
        print("Starting microservices demo...")
        
        # Check if we should start local services or connect to remote
        use_local = all(host == 'localhost' for host, port in SERVICE_INSTANCES)
        
        instances = []
        
        if use_local:
            # Start local service instances
            print("Starting local service instances...")
            for i, (host, port) in enumerate(SERVICE_INSTANCES):
                instance = ServiceInstance(f"instance_{i}", port)
                thread = threading.Thread(target=instance.start, daemon=True)
                thread.start()
                instances.append(instance)
                time.sleep(0.1)
            
            print(f"Started {len(SERVICE_INSTANCES)} local service instances")
            time.sleep(1)
        else:
            # Connect to remote services
            print(f"Connecting to {len(SERVICE_INSTANCES)} remote service instances:")
            for host, port in SERVICE_INSTANCES:
                print(f"  - {host}:{port}")
        
        # Create load balancer and add instances from SERVICE_INSTANCES
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        for i, (host, port) in enumerate(SERVICE_INSTANCES):
            info = InstanceInfo(
                instance_id=f"instance_{i}",
                host=host,
                port=port
            )
            lb.add_instance(info)
        
        # Create smart client
        client = SmartClient(lb)
        
        # Run some tests
        tester = Tester()
        tester.test_basic_functionality(client)
        tester.test_load_balancing(client, 30)
        
        # Show statistics
        print("\n=== Final Statistics ===")
        stats = lb.get_stats()
        print(json.dumps(stats, indent=2))
        
        # Cleanup local instances if any
        for instance in instances:
            instance.shutdown()
    
    elif mode == "test":
        # Run full test suite
        print("Running comprehensive test suite...")
        
        # Check if we should start local services or connect to remote
        use_local = all(host == 'localhost' for host, port in SERVICE_INSTANCES)
        
        instances = []
        
        if use_local:
            # Start local service instances
            print("Starting local service instances for testing...")
            for i, (host, port) in enumerate(SERVICE_INSTANCES):
                instance = ServiceInstance(f"test_instance_{i}", port)
                thread = threading.Thread(target=instance.start, daemon=True)
                thread.start()
                instances.append(instance)
                time.sleep(0.1)
            
            print(f"Started {len(SERVICE_INSTANCES)} local service instances")
            time.sleep(1)
        else:
            # Connect to remote services
            print(f"Testing with {len(SERVICE_INSTANCES)} remote service instances:")
            for host, port in SERVICE_INSTANCES:
                print(f"  - {host}:{port}")
            print("Make sure these services are running!")
            time.sleep(2)
        
        # Test different strategies
        strategies = [
            LoadBalancingStrategy.ROUND_ROBIN,
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.RANDOM
        ]
        
        tester = Tester()
        
        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"Testing with {strategy.value} strategy")
            print('='*50)
            
            # Create load balancer with strategy using SERVICE_INSTANCES
            lb = LoadBalancer(strategy)
            for i, (host, port) in enumerate(SERVICE_INSTANCES):
                info = InstanceInfo(
                    instance_id=f"instance_{i}",
                    host=host,
                    port=port
                )
                lb.add_instance(info)
            
            # Create client
            client = SmartClient(lb)
            
            # Run tests
            tester.test_basic_functionality(client)
            tester.test_load_balancing(client, 100)
            
            # Only run fault tolerance test with local instances
            if use_local and instances:
                tester.test_fault_tolerance(instances, client)
            else:
                print("\n=== Skipping Fault Tolerance Test (remote instances) ===")
        
        # Cleanup local instances if any
        for instance in instances:
            instance.shutdown()
        
        print("\n" + "="*50)
        print("All tests completed!")
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
