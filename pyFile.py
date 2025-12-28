import os
import subprocess
import sys

ROOT = os.path.dirname(__file__)
CPP_DIR = os.path.join(ROOT, 'cpp_source')

def build_pybind():
	# Build the pybind11 extension in-place using the current Python
	print('Building pybind11 extension (in-place)...')
	cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
	p = subprocess.run(cmd, cwd=ROOT)
	return p.returncode == 0

def run_pybind():
	try:
		import cpp_module
	except Exception as e:
		print('Failed to import cpp_module:', e)
		return 1
	# call the bound function
	cpp_module.hello_world()
	return 0

if __name__ == '__main__':
	ok = build_pybind()
	if not ok:
		print('Build failed, aborting.')
		sys.exit(1)
	rc = run_pybind()
	sys.exit(rc)
