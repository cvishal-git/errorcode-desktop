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
    // Development mode: use virtual environment Python
    const venvPythonWin = path.join(
      __dirname,
      '../../backend/venv/Scripts/python.exe',
    );
    const venvPythonUnix = path.join(
      __dirname,
      '../../backend/venv/bin/python',
    );

    if (process.platform === 'win32' && fs.existsSync(venvPythonWin)) {
      pythonPath = venvPythonWin;
    } else if (fs.existsSync(venvPythonUnix)) {
      pythonPath = venvPythonUnix;
    } else {
      pythonPath = process.platform === 'win32' ? 'python' : 'python3';
    }

    const scriptPath = path.join(__dirname, '../../backend/server.py');
    pythonArgs = [scriptPath, '--port', '48127', '--host', '127.0.0.1'];
    cwd = path.join(__dirname, '../..');
  } else {
    // Production mode: use bundled backend executable
    const backendName =
      process.platform === 'win32' ? 'backend.exe' : 'backend';
    pythonPath = path.join(process.resourcesPath, 'backend', backendName);
    pythonArgs = ['--port', '48127', '--host', '127.0.0.1'];
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
      PYTHONUNBUFFERED: '1',
    },
  });

  pythonProcess.on('error', (err) => {
    console.error('Python process error:', err);
  });

  pythonProcess.on('exit', (code, signal) => {
    console.log(`Python process exited with code ${code} and signal ${signal}`);
  });
}

async function createWindow() {
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

  // Add error logging
  mainWindow.webContents.on(
    'did-fail-load',
    (event, errorCode, errorDescription) => {
      console.error('Failed to load:', errorCode, errorDescription);
    },
  );

  mainWindow.webContents.on(
    'console-message',
    (event, level, message, line, sourceId) => {
      console.log('Console:', message);
    },
  );

  if (isDev) {
    // Try multiple ports in case 5173 is taken
    const tryPorts = [5173, 5174, 5175];
    let loaded = false;

    for (const port of tryPorts) {
      try {
        await mainWindow.loadURL(`http://localhost:${port}`);
        console.log(`✓ Loaded from port ${port}`);
        loaded = true;
        break;
      } catch (err) {
        console.log(`Port ${port} not available, trying next...`);
      }
    }

    if (!loaded) {
      console.error('Failed to connect to Vite dev server');
    }

    mainWindow.webContents.openDevTools(); // Enable for debugging
  } else {
    // In production, electron-builder packages files relative to app.getAppPath()
    console.log('=== Production Mode Debug ===');
    console.log('__dirname:', __dirname);
    console.log('app.getAppPath():', app.getAppPath());
    console.log('process.resourcesPath:', process.resourcesPath);

    // The dist folder is packaged at the app root
    const indexPath = path.join(app.getAppPath(), 'dist', 'index.html');
    console.log('Index path:', indexPath);
    console.log('Index exists:', fs.existsSync(indexPath));

    mainWindow
      .loadFile(indexPath)
      .then(() => {
        console.log('✅ Successfully loaded index.html');
      })
      .catch((err) => {
        console.error('❌ Error loading file:', err);
        // Fallback: try to load from URL if file loading fails
        mainWindow.loadURL(`file://${indexPath}`);
      });
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
