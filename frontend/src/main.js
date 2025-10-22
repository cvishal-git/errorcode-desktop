const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

const isDev = !app.isPackaged;
let mainWindow = null;
let pythonProcess = null;

function startPython() {
  const pythonPath = isDev
    ? 'python3'
    : path.join(process.resourcesPath, 'backend', 'server.exe');
  const scriptPath = isDev
    ? path.join(__dirname, '../../backend/server.py')
    : '';
  const args = isDev ? [scriptPath, '--port', '8000'] : ['--port', '8000'];

  console.log('Starting Python:', pythonPath, args);

  pythonProcess = spawn(pythonPath, args, { stdio: 'inherit' });

  pythonProcess.on('error', (err) => {
    console.error('Python error:', err);
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
