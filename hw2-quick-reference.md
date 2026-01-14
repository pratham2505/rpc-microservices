# HW2 Quick Reference Card

Print this out or keep it handy while working on the assignment!

---

## 🚀 Quick Start

### Local Testing (Weeks 1-2)
```bash
# On your laptop - open 4 terminals:
python3 rpc_assignment.py server 9000  # Terminal 1
python3 rpc_assignment.py server 9001  # Terminal 2
python3 rpc_assignment.py server 9002  # Terminal 3
python3 rpc_assignment.py test         # Terminal 4
```

### Cloud Testing (Week 3)
1. **Web:** console.cloud.google.com → Compute Engine → Create 3 VMs
2. **SSH:** Click SSH buttons → start services
3. **Edit:** Update `SERVICE_INSTANCES` in code with IPs
4. **Test:** `python3 rpc_assignment.py test` on laptop
5. **Stop:** Select VMs → click STOP button

---

## 📋 What You Need to Implement

### Part 1: RPC (30 pts)
- `handle_request()` - Process client requests
- `_calculate()` - Support sum, avg, min, max, multiply

### Part 2: Load Balancing (25 pts)
- `select_instance()` - Round-robin & least-connections
- `update_instance_stats()` - Track requests/latency

### Part 3: Fault Tolerance (25 pts)
- `send_request()` - Retry with exponential backoff
- `CircuitBreaker.call()` - Open/closed/half-open states
- `_calculate_retry_delay()` - Backoff strategy

### Part 4: Deployment (20 pts)
- Deploy to 3 GCP instances
- Test from laptop
- Take 3 screenshots

---

## 🌐 GCP Web Interface Cheat Sheet

| Task | Navigation | Action |
|------|------------|--------|
| **Create VM** | Compute Engine → VM instances | CREATE INSTANCE |
| **SSH Terminal** | VM instances list | Click SSH button |
| **Start/Stop** | VM instances list | Select → STOP/START |
| **Get IPs** | VM instances list | Look at External IP column |
| **Firewall** | VPC network → Firewall | CREATE FIREWALL RULE |
| **Check Cost** | Billing → Overview | See usage |

### Creating Instances (Quick)
- Name: `rpc-service-0` (then 1, 2)
- Region: `us-central1-a`
- Type: `e2-micro`
- OS: `Ubuntu 20.04 LTS`
- **IMPORTANT:** Check **Spot** ✓ (saves 60-80%!)

### Firewall Rule (One-Time)
- Name: `allow-rpc-services`
- TCP ports: `9000-9002`
- Source: `0.0.0.0/0`

---

## 💻 Commands for SSH Terminal

```bash
# In each VM's SSH terminal:

# 1. Clone your repo
git clone https://github.com/umb-cs446/hw2-microservices-YOURUSERNAME.git
cd hw2-microservices-YOURUSERNAME

# 2. Start service (use correct port!)
# Instance-0:
nohup python3 rpc_assignment.py server 9000 > service.log 2>&1 &

# Instance-1:
nohup python3 rpc_assignment.py server 9001 > service.log 2>&1 &

# Instance-2:
nohup python3 rpc_assignment.py server 9002 > service.log 2>&1 &

# 3. Verify running
ps aux | grep rpc_assignment
tail service.log
```

---

## 📝 Editing SERVICE_INSTANCES

In `rpc_assignment.py` on your laptop:

```python
# Comment out localhost:
# SERVICE_INSTANCES = [
#     ('localhost', 9000),
#     ('localhost', 9001),
#     ('localhost', 9002),
# ]

# Add your GCP IPs:
SERVICE_INSTANCES = [
    ('35.232.123.45', 9000),  # Your actual IPs
    ('35.232.123.46', 9001),
    ('35.232.123.47', 9002),
]
```

---

## 🎯 Testing Commands

```bash
# Local testing
python3 rpc_assignment.py test

# Demo mode
python3 rpc_assignment.py demo

# Start single server
python3 rpc_assignment.py server 9000
```

---

## 💰 Cost Management

### Expected Costs
- **Local development:** $0
- **3 cloud VMs (stopped when not using):** ~$1-2
- **Total:** < $5

### CRITICAL Commands

**STOP instances when done:**
- Web: Select all → STOP button
- CLI: `gcloud compute instances stop rpc-service-{0,1,2} --zone=us-central1-a`

**DELETE when assignment complete:**
- Web: Select all → DELETE button (trash icon)
- CLI: `gcloud compute instances delete rpc-service-{0,1,2} --zone=us-central1-a --quiet`

---

## 📸 Required Screenshots

1. **VM List:** Console → Compute Engine → instances (showing 3 running)
2. **Test Output:** Your laptop terminal showing test results

---

## 🔧 Troubleshooting

### Can't connect from laptop?
- ✓ Check firewall allows ports 9000-9002
- ✓ Check VMs are **Running** (not stopped)
- ✓ Check `SERVICE_INSTANCES` has correct IPs

### Service not running?
```bash
# SSH to VM, check process:
ps aux | grep rpc_assignment

# If not running, restart:
cd hw2-microservices-YOURUSERNAME
nohup python3 rpc_assignment.py server 9000 > service.log 2>&1 &
```

### Tests failing?
- ✓ Verify services running on all 3 VMs
- ✓ Check `SERVICE_INSTANCES` matches actual IPs
- ✓ Check firewall rule exists

---

## 📚 Key Files to Submit

- `rpc_assignment.py` - Your implementation
- `DESIGN.md` - Design decisions (1-2 pages)
- `DEPLOYMENT.md` - Screenshots + deployment notes
- `ADVANCED.md` - (Graduate students only)

---

## ⏰ Timeline

| Week | Focus | Cost |
|------|-------|------|
| 1-2 | Local development | $0 |
| 3 | Cloud deployment & testing | $2 |
| 4 | Documentation & submit | $0 |

---

## 💡 Pro Tips

✅ Test locally FIRST before deploying  
✅ Commit to GitHub frequently  
✅ Use AI tools (ChatGPT, Claude, Copilot) to help understand concepts  
✅ ALWAYS stop VMs when done for the day  
✅ Start early - distributed systems can be tricky!  


---

## ✅ Before Submitting Checklist

- [ ] All TODO sections implemented
- [ ] Tests pass locally
- [ ] Tests pass on cloud deployment
- [ ] 3 screenshots taken
- [ ] Documentation written
- [ ] Code committed and pushed to GitHub
- [ ] **VMs STOPPED or DELETED** ⚠️

---

**Keep this handy while working! Good luck! 🚀**