# launch_aquamind.py
import subprocess
import sys
import os
def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        return True
    except subprocess.CalledProcessError:
        return False
def check_and_install_dependencies():
    """Check if required packages are installed, install if missing"""
    required_packages = {
        'streamlit': 'streamlit==1.28.0',
        'pandas': 'pandas==2.1.0',
        'numpy': 'numpy==1.24.3',
        'sklearn': 'scikit-learn==1.3.0',
        'plotly': 'plotly==5.17.0'
    }

    missing_packages = []

    print("🔍 Checking dependencies...")

    for package_import, package_install in required_packages.items():
        try:
            __import__(package_import)
            print(f"  ✅ {package_import} - installed")
        except ImportError:
            print(f"  ❌ {package_import} - missing")
            missing_packages.append(package_install)

    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"  Installing {package}...")
            if install_package(package):
                print(f"  ✅ {package} installed successfully")
            else:
                print(f"  ❌ Failed to install {package}")
                return False
        print("\n✅ All dependencies installed!\n")
    else:
        print("\n✅ All dependencies already installed!\n")

    return True
def main():
    """Main launcher function"""
    print("=" * 60)
    print("💧 AquaMind Launcher")
    print("=" * 60)

    # Check and install dependencies
    if not check_and_install_dependencies():
        print("\n❌ Failed to install dependencies. Please install manually:")
        print("   pip install streamlit pandas numpy scikit-learn plotly")
        sys.exit(1)

    # Path to Streamlit app
    script_path = os.path.join(os.getcwd(), "aquamind_app.py")

    if not os.path.exists(script_path):
        print(f"❌ Error: aquamind_app.py not found in {os.getcwd()}")
        sys.exit(1)

    print(f"🚀 Launching AquaMind...")
    print(f"📂 App location: {script_path}\n")

    try:
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
    except KeyboardInterrupt:
        print("\n\n👋 AquaMind stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to launch Streamlit: {e}")
        print("\nTry running manually:")
        print(f"   streamlit run {script_path}")
        sys.exit(1)
if __name__ == "__main__":
    main()
