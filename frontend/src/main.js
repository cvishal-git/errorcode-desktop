const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const isDev = !app.isPackaged;
let mainWindow = null;
let pythonProcess = null;

function getAppRoot() {
  // In production, app root is where the .exe is located
  // In dev, it's the project root
  if (isDev) {
    return path.join(__dirname, '../..');
  }
  
  // Production: exe is in resources/app.asar or resources/app
  // We want to go to the directory containing the .exe
  return path.dirname(app.getPath('exe'));
}

function startPython() {
  const appRoot = getAppRoot();
  
  let pythonPath, pythonArgs, cwd;
  
  if (isDev) {
    // Development mode: use system Python
    pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    const scriptPath = path.join(__dirname, '../../backend/server.py');
    pythonArgs = [scriptPath, '--port', '8000', '--host', '127.0.0.1'];
    cwd = path.join(__dirname, '../..');
  } else {
    // Production mode: use bundled backend executable
    pythonPath = path.join(process.resourcesPath, 'backend', 'backend.exe');
    pythonArgs = ['--port', '8000', '--host', '127.0.0.1'];
    cwd = appRoot; // Set working directory to where models/ and database/ are
  }

  console.log('Starting Python backend...');
  console.log('  Path:', pythonPath);
  console.log('  Args:', pythonArgs);
  console.log('  CWD:', cwd);
  console.log('  App root:', appRoot);

  // Verify backend exists in production
  if (!isDev && !fs.existsSync(pythonPath)) {
    console.error('Backend executable not found at:', pythonPath);
    console.error('Make sure backend.exe is in resources/backend/');
  }

  pythonProcess = spawn(pythonPath, pythonArgs, { 
    stdio: 'inherit',
    cwd: cwd,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1'
    }
  });

  pythonProcess.on('error', (err) => {
    console.error('Python process error:', err);
  });

  pythonProcess.on('exit', (code, signal) => {
    console.log(`Python process exited with code ${code} and signal ${signal}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 450,
    height: 700,
    minWidth: 400,
    minHeight: 500,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    backgroundColor: '#1a1d29',
    titleBarStyle: 'hiddenInset',
    frame: true,
    resizable: true,
  });

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173');
    // Comment out dev tools for cleaner look
    // mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }
}

app.whenReady().then(() => {
  startPython();
  setTimeout(() => createWindow(), 2000);
});

app.on('quit', () => {
  if (pythonProcess) pythonProcess.kill();
});

app.on('window-all-closed', () => {
  app.quit();
});
