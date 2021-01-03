import os

# host & port
bind = '0.0.0.0:8000'

# Debugging
#reload = True

# Logging
accesslog = './log/access.log'
errorlog = './log/error.log'
loglevel = 'info'

# Proc Name
proc_name = 'GAN-on-flask'

# Worker Processes
workers = 1
worker_class = 'sync'
