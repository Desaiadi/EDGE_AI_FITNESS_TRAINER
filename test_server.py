#!/usr/bin/env python3
"""
Simple HTTP server to test display
"""

import http.server
import socketserver
import webbrowser
import os

def main():
    print("🌐 Starting Test Server")
    print("=" * 30)
    print("This will serve a test page to verify display works")
    
    PORT = 8000
    
    # Change to the directory containing the HTML file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"✅ Server running at http://localhost:{PORT}")
        print("Opening browser...")
        
        # Open browser
        webbrowser.open(f'http://localhost:{PORT}/test_display.html')
        
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()
