# Installing Leonardo / Instalar Leonardo

[English](#english) · [Español](#español)

---

## English

Leonardo installs everything it needs in one step: open-source PyMOL, OpenMM, AmberTools
(with PACKMOL-Memgen), Psi4, MOPAC, MODELLER, RDKit, Open Babel, smina, fpocket, BLAST,
HMMER, Clustal Omega, MAFFT and the Python packages of the AI assistant. You do not need to
install any of them yourself, and nothing is installed system-wide: everything lives in your
user folder.

**You need:** macOS (Apple Silicon or Intel) or Linux x86_64 (Ubuntu, Fedora, ...),
about **10 GB** of free disk, an internet connection, and **20–60 minutes**.

### 1. Download

Download **`Leonardo-Installer-<version>.zip`** from the release page and double-click it to
unzip it.

### 2. Install

**macOS**

1. Open the unzipped folder and double-click **`Install Leonardo (macOS).command`**.
2. The first time, macOS says it cannot verify the developer. Close that message, then
   **right-click** (or Control-click) the file, choose **Open**, and click **Open** again.
3. A Terminal window installs everything. It shows each step; just wait.

**Linux**

1. Open the unzipped folder, right-click on an empty area and choose **Open in Terminal**.
2. Type `bash install-leonardo-linux.sh` and press Return.
3. If it reports missing graphics libraries, accept installing them (it asks for your password).

### 3. MODELLER license (homology modeling)

MODELLER is free for academic use but needs a personal key:

1. Register at <https://salilab.org/modeller/registration.html>; the key arrives by e-mail.
2. Paste it when the installer asks. You can also press Return to skip it, and later open
   **Leonardo MODELLER License** (in Applications on macOS, or in the applications menu on
   Linux) and paste it there.

Everything else in Leonardo works without MODELLER.

### 4. Open Leonardo

- **macOS:** double-click **Leonardo** on the Desktop or in *Applications* (your home folder).
- **Linux:** open **Leonardo** from the applications menu or the Desktop.

PyMOL opens. Choose **Plugins → Leonardo**.

### Updating and uninstalling

- **Update:** run the installer of the new version. Your settings and your API keys are kept.
- **Uninstall:** delete the folder `~/.leonardo` and the environment
  (`~/miniforge3/bin/conda env remove -n protliginteract-full`). On macOS, also delete
  `~/Applications/Leonardo.app` and `~/Applications/Leonardo MODELLER License.app`.

### If something goes wrong

- Running the installer again is safe: it continues where it stopped.
- Every run writes a log in `~/.leonardo/install_logs/`. Send the newest one when you ask
  for help.
- PyMOL does not open on Linux: install your distribution's OpenGL/EGL packages (the
  installer prints the exact command).

### For developers

From a clone of the repository: `bash installers/install_leonardo.sh` installs the environment
and points Leonardo at your working copy. When the repository is public, users can also run
`curl -fsSL https://github.com/juvenalyosa/Leonardo/releases/download/<tag>/install_leonardo.sh | bash`.
Options: `--yes` (unattended), `--modeller-key KEY`, `--skip-env`, `--no-shortcuts`, `--help`.

---

## Español

Leonardo instala todo lo que necesita en un solo paso: PyMOL de código abierto, OpenMM,
AmberTools (con PACKMOL-Memgen), Psi4, MOPAC, MODELLER, RDKit, Open Babel, smina, fpocket,
BLAST, HMMER, Clustal Omega, MAFFT y los paquetes de Python del asistente de IA. No tiene que
instalar ninguno a mano, y nada se instala en el sistema: todo queda en su carpeta de usuario.

**Necesita:** macOS (Apple Silicon o Intel) o Linux x86_64 (Ubuntu, Fedora, ...),
unos **10 GB** libres, conexión a internet y **20–60 minutos**.

### 1. Descargar

Descargue **`Leonardo-Installer-<versión>.zip`** de la página de la versión y haga doble clic
para descomprimirlo.

### 2. Instalar

**macOS**

1. Abra la carpeta descomprimida y haga doble clic en **`Install Leonardo (macOS).command`**.
2. La primera vez macOS dice que no puede verificar al desarrollador. Cierre ese mensaje, haga
   **clic derecho** (o Control-clic) sobre el archivo, elija **Abrir** y pulse **Abrir** otra vez.
3. Una ventana de Terminal instala todo y muestra cada paso; solo espere.

**Linux**

1. Abra la carpeta descomprimida, haga clic derecho en un espacio vacío y elija
   **Abrir en terminal**.
2. Escriba `bash install-leonardo-linux.sh` y pulse Enter.
3. Si avisa que faltan librerías gráficas, acepte instalarlas (le pedirá su contraseña).

### 3. Licencia de MODELLER (modelado por homología)

MODELLER es gratuito para uso académico, pero necesita una clave personal:

1. Regístrese en <https://salilab.org/modeller/registration.html>; la clave llega por correo.
2. Péguela cuando el instalador la pida. También puede pulsar Enter para saltarla, y después
   abrir **Leonardo MODELLER License** (en Aplicaciones en macOS, o en el menú de aplicaciones
   en Linux) y pegarla allí.

Todo lo demás de Leonardo funciona sin MODELLER.

### 4. Abrir Leonardo

- **macOS:** doble clic en **Leonardo** en el Escritorio o en *Aplicaciones* (de su carpeta de
  usuario).
- **Linux:** abra **Leonardo** desde el menú de aplicaciones o el Escritorio.

Se abre PyMOL. Elija **Plugins → Leonardo**.

### Actualizar y desinstalar

- **Actualizar:** ejecute el instalador de la nueva versión. Se conservan su configuración y
  sus claves de API.
- **Desinstalar:** borre la carpeta `~/.leonardo` y el entorno
  (`~/miniforge3/bin/conda env remove -n protliginteract-full`). En macOS borre también
  `~/Applications/Leonardo.app` y `~/Applications/Leonardo MODELLER License.app`.

### Si algo falla

- Volver a ejecutar el instalador es seguro: continúa donde se detuvo.
- Cada ejecución deja un registro en `~/.leonardo/install_logs/`. Envíe el más reciente cuando
  pida ayuda.
- Si PyMOL no abre en Linux, instale los paquetes OpenGL/EGL de su distribución (el instalador
  muestra el comando exacto).
