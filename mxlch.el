;;; mxlch.el --- Helpful functions for running MXLCH)

;; Copyright (C) Adam Massmann

;; Author: Adam Massmann <akm2203@columbia.edu>
;; Created: 6 Jan 2022
;; Version: 0.0.0
;; Keywords: climate
;; Package-Requires: ((emacs "26.1") (cl-lib "1.0"))

;;; Commentary:

;; See https://github.com/massma/land-atmosphere for further
;; information.


;;; Code:


;;;; mxlch
(defcustom mxlch-dir (concat (getenv "HOME") "/class/mxlch")
  "The directory to find the MXLCH repository.  See https://github.com/classmodel/mxlch."
  :type 'string
  :group 'mxlch)

(defcustom mxlch-repo-dir (concat (getenv "HOME") "/land-atmosphere")
  "The directory for the land-atmosphere repository.

TODO: could probably set this with introspection, because it is wherever
this file is!"
  :type 'string
  :group 'mxlch)

(defcustom mxlch-data-dir (concat (getenv "HOME") "/land-atmosphere/data")
  "The directory to find MXLCH data and simulations, as output by soundings.py."
  :type 'string
  :group 'mxlch)

(defconst mxlch-variable-input "variables.el"
  "The filename of variable input.")

(defconst mxlch-sites
  '("elko"
    "bergen"
    "flagstaff"
    "great_falls"
    "idar_oberstein"
    "kelowna"
    "las_vegas"
    "lindenberg"
    "milano"
    "quad_city"
    "riverton"
    "spokane"))


;;;;  Utility functions
(defun mxlch-generic-quarter (latlon f)
  "Caluclate the nearest (as defined by F) quarter degree of LATLON.

F can be something like round, ceiling, floor."
  (number-to-string
   (+ 0.25 (* 0.5 (funcall f (* 2.0 (- latlon 0.25)))))))

(defun mxlch-move-sites ()
  "One-off utility function to move sites.  Keeping as an example."
  (dolist (site mxlch-sites)
    (rename-file (concat mxlch-data-dir "/" site "-reality-slope.csv")
                 (concat mxlch-data-dir "/" site "-realistic.csv")
                 nil)
    (rename-file (concat mxlch-data-dir "/" site "-reality-slope")
                 (concat mxlch-data-dir "/" site "-realistic")
                 nil)
    (rename-file (concat mxlch-data-dir "/" site "-randomized.csv")
                 (concat mxlch-data-dir "/" site "-deconfounded.csv")
                 nil)
    (rename-file (concat mxlch-data-dir "/" site "-randomized")
                 (concat mxlch-data-dir "/" site "-deconfounded")
                 nil)))

(defun mxlch-floor-quarter (latlon)
  "Calculate the nearest quarter degree of LATLON."
  (mxlch-generic-quarter latlon #'floor))

(defun mxlch-ceiling-quarter (latlon)
  "Calculate the nearest quarter degree of LATLON."
  (mxlch-generic-quarter latlon #'ceiling))

(defun mxlch-nearest-quarter (latlon)
  "Calculate the nearest quarter degree of LATLON."
  (mxlch-generic-quarter latlon #'round))

(defun mxlch-lat-lon-search (lat lon)
  "Search buffer for LAT and LON, separated by whitespace."
  (re-search-forward (rx bol
                         (* whitespace) (literal lat)
                         (* whitespace) (literal lon))))

(defun mxlch-find-nearest-koppen (lat lon)
  "Search the Koppen climate file for the 4 points nearest to LAT and LON."
  (interactive "nLat: \nnLon: ")
  (save-window-excursion
    (let* ((closest-lat (mxlch-nearest-quarter lat))
           (smallest-lat (mxlch-floor-quarter lat))
           (largest-lat (mxlch-ceiling-quarter lat))
           (closest-lon (mxlch-nearest-quarter lon))
           (smallest-lon (mxlch-floor-quarter lon))
           (largest-lon (mxlch-ceiling-quarter lon))
           (output-str "")
           (f (lambda (lat lon id-str)
                (let ((nearest-str (if (and (string= lat closest-lat)
                              (string= lon closest-lon))
                         "(nearest)"
                       "")))
                  (goto-char (point-min))
                  (mxlch-lat-lon-search lat lon)
                  (setq output-str
                        (concat output-str
                                (format "%s corner: %s %s\n"
                                        id-str
                                        (substring (thing-at-point 'line t)
                                                   0 -1)
                                        nearest-str)))))))
      (find-file (concat mxlch-data-dir "/" "1976-2000_ASCII.txt.gz"))
      (funcall f smallest-lat smallest-lon "SW")
      (funcall f largest-lat smallest-lon "NW")
      (funcall f largest-lat largest-lon "NE")
      (funcall f smallest-lat largest-lon "SE")
      (kill-new output-str))))

(defun mxlch-generate-defvar ()
  "Make a namelist file into a bunch of defvars, and add it to the kill ring."
  (interactive)
  (let ((buf (get-buffer-create "*mxlch-generate-defvar*"))
        (contents (buffer-string)))
    (save-mark-and-excursion
      (with-current-buffer buf
        (erase-buffer)
        (insert contents)
        (goto-char (point-min))
        (while

            (re-search-forward
             (rx "
" (group (+ (any alnum punct))) (+ blank) "=" (+ blank)
(group (+ (any alnum punct))) (* not-newline) "
")
             nil
             t)
          (replace-match "
(defvar mxlch-\\1 \"\\2\")
")
          (backward-char))
        (goto-char (point-min))
        (while (re-search-forward
                (rx "&NAM" (group (+ upper-case)))
                nil
                t)
          (replace-match ";; NAM\\1"))
        (goto-char (point-min))
        (while (re-search-forward
                (rx "
/
") nil t)
          (replace-match "
"))
        (kill-ring-save (point-min) (point-max))))))

(defun mxlch-parse-namelist ()
  "Parse a namelist file and add it to the kill ring."
  (interactive)
  (let ((buf (get-buffer-create "*mxlch-generate-defvar*"))
        (contents (buffer-string)))
    (save-mark-and-excursion
      (with-current-buffer buf
        (erase-buffer)
        (insert contents)
        (goto-char (point-min))
        (re-search-forward (rx "namelist/" (group (+ upper)) "/" (* blank) "&"))
        (while
            (re-search-forward
             (rx "
" (* blank) (group (+? (any alnum punct))) (* blank) ",&"  (* blank) "!" (group (+ not-newline)))
             nil
             t)
          (replace-match "
(defvar mxlch-\\1 \"\\2\")"))
        (kill-ring-save (point-min) (point-max))))))

(defun mxlch-write-variable (name)
  "Write variable with name (a string) to the namelist."
  (eval `(insert ,`(format "%-15s = %s
" ,name ,(intern (concat "mxlch-" name))))))

(defun mxlch-write-block (name list)
  "Write a namelist block of NAME, with LIST of variables."
  (insert "&NAM" name "
")
  (dolist (var list)
    (mxlch-write-variable (symbol-name var)))
  (insert "/

"))

(defun mxlch-write-namelist (&optional filename)
  "Write the current state to a namelist in FILENAME."
  (if filename
      (find-file filename)
    (ido-file-internal ido-default-file-method))
  (erase-buffer)
  (mxlch-write-block
   "RUN"
   '(outdir time dtime atime atime_vert h_max latt long day hour))
  (mxlch-write-block
   "DYN"
   '(zi0 beta lenhancedentrainment wsls lfixedlapserates
         wthetasmax c_fluxes gamma thetam0 dtheta0 advtheta
         pressure wqsmax gammaq qm0 dq0 advq wcsmax gammac cm0 dc0 c_ustr
         z0 uws0 vws0 gammau gammav um0 vm0 ug vg lencroachment lscu lrelaxdz
         tau ladvecFT))
  (mxlch-write-block
   "SURFLAYER"
   '(lsurfacelayer z0m z0h))
  (mxlch-write-block
   "RAD"
   '(lradiation cc S0 albedo))
  (mxlch-write-block
   "SURFACE"
   '(llandsurface Qtot lsea sst Ts wwilt w2 wg wfc wsat CLa CLb CLc C1sat
                  C2ref gD rsmin rssoilmin LAI cveg Tsoil T2 Wl Lambda CGsat lrsAgs
                  lCO2Ags CO2comp298 Q10CO2 gm298 Ammax298 Q10gm T1gm T2gm Q10Am
                  T1Am T2Am f0 ad alpha0 Kx gmin Cw wsmax wsmin R10 Eact0 lBVOC
                  BaserateIso BaserateTer))
  (mxlch-write-block
   "CHEM"
   '(lchem lcomplex lwritepl ldiuvar h_ref lflux fluxstart fluxend pressure_ft lchconst
           t_ref_cbl p_ref_cbl q_ref_cbl t_ref_ft p_ref_ft q_ref_ft))
  (mxlch-write-block
   "FLUX"
   '(offset_wt offset_wq function_wt function_wq))
  (mxlch-write-block
   "SOA"
   '(lvbs low_high_NOx alpha1_TERP_low alpha2_TERP_low alpha3_TERP_low
          alpha4_TERP_low alpha1_TERP_high alpha2_TERP_high alpha3_TERP_high
          alpha4_TERP_high alpha1_ISO_low alpha2_ISO_low alpha3_ISO_low
          alpha1_ISO_high alpha2_ISO_high alpha3_ISO_high))
  (save-buffer)
  (kill-buffer))


(defun mxlch-write-namelists-experiment (exp-name arg)
  "Write out all namelists for experiment EXP-NAME.

If ARG is true, overwrite exisitng nameslist if it exists."
  (let ((inputs (directory-files-recursively
                 mxlch-data-dir
                 (rx (literal mxlch-variable-input))
                 nil
                 (lambda (dir)
                   (string-match (rx (literal exp-name) (* anything)) dir)))))
    (dolist (input inputs)
      (let ((namelist-path (concat (file-name-directory input)
                                   "namoptions")))
        (when (or arg
                  (not (file-exists-p namelist-path)))
          (load input)
          (mxlch-write-namelist namelist-path))))
    't))

(defun mxlch-site-name-from-path (path)
  "Get the experiment name from PATH."
  (string-match (rx (group (+? (not blank))) "-constants.el") path)
  (match-string 1 path))

(defun mxlch-write-constants ()
  "Write all MXLCH constant values to a csv file."
  (interactive)
  (mxlch-set-non-default-constants)
  (let ((constant-inputs (directory-files
                          mxlch-data-dir
                          nil
                          (rx (+? (not blank)) "-constants.el"))))
    (find-file (concat mxlch-data-dir "/" "site-constants.csv"))
    (erase-buffer)
    (insert "site,C1sat,C2ref,CLa,CLb,CLc,albedo,cveg,latt,wfc,wsat,wwilt,z0h,z0m\n")
    (dolist (input constant-inputs)
      (load (concat mxlch-data-dir "/" input))
      (insert (mxlch-site-name-from-path input))
      (insert ",")
      (dolist (x (list mxlch-C1sat mxlch-C2ref mxlch-CLa mxlch-CLb
                       mxlch-CLc mxlch-albedo mxlch-cveg
                       mxlch-latt mxlch-wfc mxlch-wsat
                       mxlch-wwilt mxlch-z0h mxlch-z0m))
        (insert x) (insert ","))
      (delete-char -1)
      (insert "
"))
    (save-buffer)
    (kill-buffer)))


(defun mxlch-write-namelists (arg)
  "Write all namelists for experiments in `mxlch-data-dir'.

Will do nothing if a namelist already exists,
 unless prefix argument ARG is positive.

In that case, it will overwrite the namelist."
  (interactive "P")
  (mxlch-set-non-default-constants)
  (let ((constant-inputs (directory-files
                          mxlch-data-dir
                          nil
                          (rx (+? (not blank)) "-constants.el"))))
    (dolist (input constant-inputs)
      (load (concat mxlch-data-dir "/" input))
      (mxlch-write-namelists-experiment
       (mxlch-site-name-from-path input)
       arg))))

(defun mxlch-run-models (arg)
  "Run all models for experiments in `mxlch-data-dir'.

Will only run a model if there is no \"model-run.out file\", AND
the prefix ARG is false.

This could be asnychronous, but we end up with too many pipes open.

Would have to use a more sophisticated handler using sentinels in that case."
  (interactive "P")
  (let ((inputs (directory-files-recursively
                 mxlch-data-dir
                 (rx  (*? anything)
                      (literal mxlch-variable-input)))))
    (dolist (input inputs)
      (let* ((default-directory  (file-name-directory input))
             (output-log (concat default-directory "/model-run.out")))
        (when (or arg
                  (not (file-exists-p output-log)))
          (call-process "rm" nil nil nil "-rf" (concat default-directory "RUN00") "model-run.out")
          (call-process
           (concat mxlch-dir "/MXLCH")
           nil
           `(:file ,(concat default-directory "/model-run.out"))
           nil
           (concat mxlch-dir "/chem.inp.op3")))))
    't))

(defun mxlch-extract-et (dir)
  "Extract et from output_land file in experiment DIR.

Currently, this takes all data between 1 (inclusive) and 17 (exclusive)
local time, and averages it.

It returns ET as a string, and if it encounters any NaN's,
it will return \"NaN\"."
  (find-file (concat dir "/RUN00/output_land"))
  (goto-char (point-min))
  (forward-line 3)
  (move-beginning-of-line 1)
  (let ((time)
        (le)
        (le-1 nil)
        (lw-1 nil)
        (lw nil)
        (sh nil)
        (sh-1 nil)
        (break nil)
        (count 0)
        (sum 0.0)
        (search (lambda ()
                  (re-search-forward (rx (* blank)
                                         (group (or "NaN" (+ (or num ?.)))))
                                     (line-end-position)
                                     t))))
    (while (and
            (re-search-forward (rx (* blank) (group (+ (or num ?.))))
                               (line-end-position)
                               t)
            (numberp sum)
            (not break))
      (setq time (read (match-string 1)))
      (dotimes (_x 5) (funcall search))
      (setq lw (read (match-string 1)))
      (unless lw-1
        (setq lw-1 lw))
      (dotimes (_x 2) (funcall search))
      (setq sh (read (match-string 1)))
      (unless sh-1
        (setq sh-1 sh))
      (funcall search)
      (setq le (read (match-string 1)))
      (unless le-1
        (setq le-1 le))
      (cond
       ((eq 'NaN lw) (setq sum 'NaN))
       ((eq 'NaN sh) (setq sum 'NaN))
       ((eq 'NaN le) (setq sum 'NaN))
       ((> (abs (- lw lw-1)) 200) (setq sum 'NaN))
       ((> (abs (- sh sh-1)) 900) (setq sum 'NaN))
       ((> (abs (- le le-1)) 450) (setq sum 'NaN))
       ((and (>= time 13.0)
             (< time 17.0))
        (setq count (+ count 1))
        (setq sum (+ sum le)))
       ((> time 17.0) (setq break t))
       (t t ;; time < 13, do nothing
          ))
      (setq le-1 le)
      (setq lw-1 lw)
      (setq sh-1 sh)
      (move-beginning-of-line 2))
    (kill-buffer)
    (cond
     ((eq 'NaN sum) "nan")
     ((= 0 count) "nan")
     (t (number-to-string (/ sum (float count)))))))

(defun mxlch-sample (exp-path)
  "Get sample from EXP-PATH.

Returns nil if EXP-PATH does not contain a sample number."
  (if
      (string-match (rx bos (group (+? (or ?_ alpha))) ?_ (group (+ digit)) eos)
                    exp-path)
      (list
       (match-string 1 exp-path)
       (number-to-string (string-to-number
                          (match-string 2 exp-path))))
    nil))

(defun mxlch-year-doy-experiment (exp-path)
  "Get year, doy, and experiment from EXP-PATH.

Returns nil if EXP-PATH does not contain those data."
  (if
      (string-match (rx bos
                        (group (+? (or ?_ alpha))) ?_
                        (group (+ digit)) ?_
                        (group (+ digit)) "_SM"
                        (group (+ (or ?- digit)))
                        eos)
                    exp-path)
      (list
       (match-string 1 exp-path)
       (number-to-string (string-to-number
                          (match-string 2 exp-path)))
       (number-to-string (string-to-number
                          (match-string 3 exp-path)))
       (number-to-string (string-to-number
                          (match-string 4 exp-path))))
    nil))

(defun mxlch-load-output (input)
  "Load model output corresponding to input file INPUT.

The returned data structure will be a list of length 3:
\(experiment name [string], soil moisture [string], ET [string]."
  (load-file input)
  (let* ((dir (file-name-directory input))
         (et (mxlch-extract-et dir))
         (base-name (file-name-base (directory-file-name dir)))
         (maybe-sample (mxlch-sample base-name))
         (maybe-metadata (mxlch-year-doy-experiment base-name))
         site sample year doy experiment)
    (cond
     (maybe-sample (setq site (car maybe-sample))
                   (setq sample (cadr maybe-sample))
                   (setq year "-9999")
                   (setq doy "-9999")
                   (setq experiment "-9999"))
     (maybe-metadata (setq site (car maybe-metadata))
                     (setq year (cadr maybe-metadata))
                     (setq doy (caddr maybe-metadata))
                     (setq experiment (cadddr maybe-metadata))
                     (setq sample "-9999"))
     (t (error (format "When writing csv, could not get a sample or day of year from %s path"
                       base-name))))
    (list site
          sample
          year
          doy
          experiment
          mxlch-wg
          et
          mxlch-T2
          mxlch-Tsoil
          mxlch-Ts
          mxlch-thetam0
          mxlch-advtheta
          mxlch-qm0
          mxlch-advq
          mxlch-LAI
          mxlch-cc
          mxlch-um0
          mxlch-vm0
          mxlch-zi0
          mxlch-pressure
          mxlch-day
          mxlch-hour
          mxlch-time)))


(defun mxlch-csv-file-name (dir)
  "Generate a csv filename from experiment directory DIR."
  (concat (directory-file-name dir) ".csv"))

(defun mxlch-write-csv (dir)
  "Write a csv from all experiments in DIR."
  (let ((filename (mxlch-csv-file-name dir))
        (inputs (directory-files-recursively
                 dir
                 (rx  (*? anything)
                      (literal
                       mxlch-variable-input)))))
    (find-file filename)
    (erase-buffer)
    (insert "site,sample-n,year,doy,experiment,SM,ET,T2,Tsoil,Ts,theta,advtheta,q,advq,LAI,cc,u,v,h,pressure,day,tstart,runtime
")
    (dolist (input inputs)
      (dolist (value (mxlch-load-output input))
        (insert value)
        (insert ","))
      (delete-char -1)
      (insert "
"))
    (save-buffer)
    (kill-buffer)))

(defun mxlch-write-all-csvs (arg)
  "Write all model output csvs.

Call `mxlch-write-csv' on every directory in `mxlch-data-dir'.

This will only be called if prefix ARG is true, OR there is no existing CSV file."
  (interactive "P")
  (let ((exp-dirs (directory-files mxlch-data-dir
                                   t
                                   (rx string-start (+? (not ?.)) string-end))))
    (dolist (dir exp-dirs)
      (when (and (file-directory-p dir)
                 (or arg
                     (not (file-exists-p (mxlch-csv-file-name dir)))))
          (mxlch-write-csv dir)))))

(defun mxlch-run-analysis (arg)
  "Run all analysis functions, passing prefix ARG to each ones.

Currently, this invovles:

1. calling python3 soundings.py
2. `mxlch-write-nameslists'
3. `mxlch-run-models'
4. `mxlch-write-all-csvs'.

Basically, call this function with a prefix argument if you want to rerun
all analyses and rewrite output data.

But be wary, this will grab and lock up your emacs."
  (interactive "P")
  (let ((default-directory mxlch-repo-dir))
    (call-process "python3"
                  nil
                  '(:file "soundings.out")
                  nil
                  "soundings.py"))
  (mxlch-write-namelists arg)
  (mxlch-write-constants)
  (mxlch-run-models arg)
  (mxlch-write-all-csvs arg)
  'ok)


;;;; Name options
;; NAMRUN
(defcustom mxlch-outdir "'RUN00'" "Name of output folder. Must be enclosed in \"'\""
  :type 'string
  :group 'mxlch)
(defcustom mxlch-time "86400" "Simulated time in seconds."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-dtime "1" "Timestep in seconds."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-atime "60" "Time interval for statistics in seconds."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-atime_vert "1800" "Time interval for vertical profile statistics in seconds."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-h_max "3000" "Maximum height of the simulated domain in meters."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-latt "0" "Latitude of simulated location in degrees.
Should be between -90 and 90 degrees, inclusive."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-long "0" "Longitude of simulated location in degrees.
Must be between 0 and 360 degrees, inclusive."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-day "80" "Julian day of the year.
Between 1 and 365."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-hour "0" "Local time at which the simulation starts in hours.
Between 0 and 24 hours."
  :type 'string
  :group 'mxlch)

;; NAMDYN
(defcustom mxlch-zi0 "200.0" "Initital boundary layer height in meters."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-beta "0.20" "Entrainment ratio."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lenhancedentrainment ".false."
  "If true, the entrainment is enhanced by wind shear (u_{star})."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wsls "0.000" "Flow divergence factor for subsidence, in s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lfixedlapserates ".false."
  "If true, the (enhancing) effect of subsidence on free tropospheric gradients is omitted."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lfixedtroposphere ".false."
  "UNDOCUMENTED in pdf."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wthetasmax "0.0"
  "Maximum surface kinematic heat flux for standard flux profiles in K m s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-c_fluxes ".false."
  "If true all fluxes are constant. It is better to use the options in NAMFLUX.

(also note that this replace c_wth option in the NAMELIST."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gamma "0.006"
  "Potential temperature lapse rate in the free troposphere in K m^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lgamma ".false."
  "A switch to apply a different lapse rate (`mxlch-gamma2') to heights above some hcrit.

TODO: undocumented in PDF; I could add."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-hcrit "10000"
  "When `mxlch-lgamma' is true, apply `mxlch-gamma2' when zi > `mxlch-hcrit'.

TODO: undocumented in PDF: I could add."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gamma2 "0.006"
  "When `mxlch-lgamma' is true, apply this lapse rate when zi > hcrit.

TODO: undocumented in PDF: I could add."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-thetam0 "295.0"
  "Initial mixed layer potential temperature in K."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-dtheta0 "4"
  "Initital potential temperature jump in K."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-pressure "1013.0"
  "Air pressure in the boundary layer in hPa. TODO:fix pdf documentation with wrong units!"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wqsmax "0.00"
  "Maximum surface kinematic moisture flux for standard flux profiles, in g kg^{-1} m s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gammaq "0.0"
  "Specific humidity lapse rate in the free troposphere in g kg^{-1} m^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-qm0 "0"
  "Initial mixed layer specific humidity in g kg^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-dq0 "0"
  "Inititial specific humidity jump in g kg^{-1}.

Must be greater than or equal to  -`mxlch-qm0'."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wcsmax "0.0"
  "Maximum surface kinematic tracer flux for standard flub proiles in ppb m s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gammac "0.0"
  "Tracer lapse rate in the free troposphere in ppb m^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-cm0 "0.0"
  "Initital mixed layer tracer concentration in ppb.
\(confirmed, I think, in code that this is in fact ppb and NOT ppm)

I think the model interprets this as carbon dioxide; see note in .f90:

\"! cm0: initial carbon dioxide mixing layer value
! dc0: initial carbon dioxide jump
\""
  :type 'string
  :group 'mxlch)
(defcustom mxlch-dc0 "0.0"
  "Initial tracer concentration part in ppb.
\(confirmed, I think, in code that this is in fact ppb and NOT ppm)

Must be greater than or equal to -`mxlch-cm0'

I think the model interprets this as carbon dioxide.

See note in `mxlch-cm0' docs."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-c_ustr ".true."
  "If true the momentum fluxes (and friction velocity) are constant.

TODO: change in code to be lc_ustr to match coding convention on logical variables?

This would make a backwards incompatable change to the namelist though, which
might not be ideal."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-z0 "0.03"
  "Roughness length in m.

TODO: Remove misleading comment in the fortran files falsly
stating that z0 is \"initial boundary layer height\"."
  :type 'string
  :group 'mxlch)

(defcustom mxlch-uws0 "0"
  "Initital surface (x-)momentum flux in m^2 s^{-2}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-vws0 "0"
  "Initial surface (y-)momentum flux in m^2 s^{-2}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gammau "0"
  "Lapse rate of u in the free troposphere in s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gammav "0"
  "Lapse rate of of v in the free troposphere in s ^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-um0 "0"
  "Initital u in the mixed layer in m s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-vm0 "0"
  "Initital v in the mixed layer in m s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-ug "0"
  "Geostrophic wind in the x-direction in m s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-vg "0"
  "Geostrophic wind in the y-direction in m s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-advq "0.0"
  "Advection of specific humidity. g kg^{-1} s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-advtheta "0.0"
  "Advection of potential temperature in K s^{-1}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-ladvecFT ".false." "If true advection is also applied for free troposphere."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lencroachment ".false." "Enables encroachment"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lscu ".false." "Enables shallow cumulus mass-flux parameterization."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lrelaxdz ".false." "If true, LCL-z_i is nudged to calculated value by time-scale \tau, rather than being set to that value."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-tau "7200" "Time-scale for nudging transition layer depth in seconds."
  :type 'string
  :group 'mxlch)


;; NAMSURFLAYER
(defcustom mxlch-lsurfacelayer ".false." "Enable or disable surface layer."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-z0m "0.03" "Roughness length of momentum in m.

Can get good defaults from the landsoil.cpp file in CLASS."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-z0h "0.03" "Roughness lenth of heat in m.

Can get good defaults from the landsoil.cpp file in CLASS."
  :type 'string
  :group 'mxlch)

;; NAMRAD
(defcustom mxlch-lradiation ".false." "Enable or disable
radiation. If this is turned off, net surface radiation is just a
constant."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-cc "0" "Amount of cloud cover; unitless between 0 and 1."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-S0 "1368" "Incoming shortwave solar radiation in W m^{-2}."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-DeltaFsw "0.0" "Absorbed radiation by e.g. aerosols (neg. value) (shortwave component). [UNDOCUMENTED]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-DeltaFlw "0.0" "Emitted radiation by e.g. clouds (pos. value) (longwave compoonent). [UNDOCUMENTED]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Rdistr "1.0" "Distribution of absorbing aerosols (see Barbero et al., 2013)"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-albedo "0.2" "Albedo as a fraction between 0 and 1."
  :type 'string
  :group 'mxlch)

;; NAMSURFACE
(defcustom mxlch-llandsurface ".false." "switch to use interactive landsurface"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Qtot "400" "Incoming energy in W m^{-2}"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lsea ".false." "Using a sea surface instead of land."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-sst "285" "Sea surface temperature in Kelvin."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Ts "295.0"
  "Initial surface temperature [K].

Note in MXLCH this defaults to `mxlch-thetam0', however in the
.el interface it will always be set to this value."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wwilt "0.314" "Wilting point (m^3 m^{-3})."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-w2 "0.420" "Volumetric water content deeper soil layer. (m^3 m^{-3})"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wg "0.40" "Volumetric water content top soil layer (m^3 m^{-3})."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wfc "0.491" "Volumetric water content field capacity (m^3 m^{-3})."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wsat "0.6" "Saturated volumetric water content ECMWF config (m^3 m^{-3})."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-CLa "0.083" "Clapp and Hornberger retention curve parameter a [unitless]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-CLb "11.4" "Clapp and Hornberger retention curve parameter b [unitless]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-CLc "12.0" "Clapp and Hornberger retention curve
parameter c [unitless] (this is parameter \"p\" in textbook, soo
page 140)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-C1sat "0.342" "Coefficient force term moisture (for heat, see page 140 in textbook)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-C2ref "0.3" "Coefficient restore term moisture (for heat, see page 140 in textbook)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gD "0.0" "VPD correction factor for rs TODO:units?. This is only used of lrsAgs is false. Also the formulation for rs doesn't make sense to me, something to check."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-rsmin "0.0" "Minimum resistance of transpiration (s m^{-1}).

This is only used of lrsAgs is false. Also the formulation for rs
doesn't make sense to me, something to check. Basically if rsmin
is 0.0, then rs will be 0.0 independent of all the other factors
and vpd, etc."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-rssoilmin "0.0" "Minimum resistance of soiltranspiration [s m^{-1}].

It seems like this formulation is broken: if rssoilmin is 0.0, then it will always be 0.0 independent of the other factors. look up Jarvis-Steward stomatal conudctance model."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-LAI "1.0" "Leaf area index [m^2 m^{-2}]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-cveg "1.0" "Vegetation fraction"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Tsoil "285" "Temperature top soil layer [K]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-T2 "285" "Temperature deeper soil layer [K]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Wl "0.0" "Equivalent water layer depth for wet vegetation [m]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Lambda "5.9" "Thermal diffusivity skin layer [W m^{-2} K^{-1}]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-CGsat "3.6e-6" "Saturated soil conductivity for heat [K m^{2} J^{-1}]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lrsAgs ".false." "Switch to use A-gs model for surface resistances"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lCO2Ags ".false." "Switch to use A-gs model for CO2 flux"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-CO2comp298 "68.5" "CO2 compensation concentration [mg m-3].

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Q10CO2 "1.5" "function parameter to calculate CO2 compensation concentration [-]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gm298 "7" "mesophyill conductance at 298 K [mm s-1]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Ammax298 "2.2" "CO2 maximal primary productivity [mg m-2 s-1]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Q10gm "2" "function parameter to calculate mesophyll conductance [-]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-T1gm "278" "reference temperature to calculate mesophyll conductance gm [K]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-T2gm "301" "reference temperature to calculate mesophyll conductance gm [K]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Q10Am "2" "function parameter to calculate maximal primary profuctivity Ammax

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-T1Am "281" "reference temperature to calculate maximal primary profuctivity Ammax [K]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-T2Am "311" "reference temperature to calculate maximal primary profuctivity Ammax [K]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-f0 "0.89" "maximum value Cfrac [-]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-ad "0.07" "regression coefficient to calculate Cfrac [kPa^{-1}]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha0 "0.017" "initial low light conditions [mg J-1]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Kx "0.7" "extinction coefficient PAR [-]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-gmin "2.5e-4" "cuticular (minimum) conductance [m s-1]

Matches for c3 plants in the CLASS model (see model.cpp)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Cw "1.6e-3" "constant water stress correction (eq. 13 Jacobs et al. 2007) [-]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wsmax "0.55" "upper reference value soil water [-]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-wsmin "0.005" "lower reference value soil water [-]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-R10 "0.23" "respiration at 10 C [mg CO2 m-2 s-1]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-Eact0 "53.3e3" "activation energy [53.3 kJ kmol-1]"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lBVOC ".false." "Enable the calculation of BVOC (isoprene, terpene) emissions.

UNDOCUMENTED in pdf."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-BaserateIso "0.0" "Base emission rate for isoprene emissions [microg m^4 h^-1].

UNDOCUMENTED in pdf."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-BaserateTer "0.0" "Base emission rate for terprene emissions [microg m^4 h^-1].

UNDOCUMENTED in pdf."
  :type 'string
  :group 'mxlch)

;; NAMCHEM
(defcustom mxlch-lchem ".false." "Enable or disable chemistry."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lwritepl ".true." "Enable the output of production and loss terms per chemical."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lcomplex ".false." "Choice between complex chemical scheme and simplified scheme."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-ldiuvar ".true." "If false the UC radiation during day is calculated at time h.ref."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-h_ref "12." "Reference time for calculated UC radiation if ldiuvar is set to .false."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lflux ".false." "If set to .true. the times of sunrise and sunset are input. The otpions in NAMFLUX are preferred."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-fluxstart "0.0" "Time of sunrise if lflux is set to .true. [hr]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-fluxend "0.0" "Time of sunset if lflux is set to .true. [hr]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-pressure_ft "1013.0"
  "Defaults to pressure [hPa] in fortran code.

TODO: fix pdf documentation to match units"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-lchconst ".false."
  "Switch to calculate reaction rates using reference
  temperatures, humidities, and pressures instead of actual
  values."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-t_ref_cbl "298."
  "Reference temperature in the boundary layer [K]."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-p_ref_cbl "1013.5"
  "Reference pressure in the boundary layer [hPa].

TODO: pdf reference has the wrong default (1000.0) and wrong units.!"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-q_ref_cbl "10.0"
  "Reference specific humidity in the boundary layer [g kg-1].

"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-t_ref_ft "298."
  "Reference temeprature in the free troposphere [K].

"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-p_ref_ft "1013.5" "Reference pressure in the free
troposphere [hPa]

I do not think tihs actually gets used, but I could be wrong..

TODO: pdf reference has the wrong default (1000.0) and wrong units.!"
  :type 'string
  :group 'mxlch)
(defcustom mxlch-q_ref_ft "10.0"
  "Reference specific humidty in the
free troposphere. [g kg-1]."
  :type 'string
  :group 'mxlch)

;; NAMFLUX
(defcustom mxlch-starttime_wt "sunrise"
  "Time after which the heat flux starts in the case of functions 2 and 3 [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-endtime_wt "sunset"
  "Time after which the heat flux ends in the case of functions 2 and 3 [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-offset_wt "0" "Offset for the kinematic heat flux [K m s-1]."
  :type 'string
  :group 'mxlch)

(defcustom mxlch-starttime_wq "sunrise"
  "Time after which the moisture flux starts in the case of functions 2 and 3 [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-endtime_wq "sunset"
  "Time after which the moisture flux ends in the case of functions 2 and 3 [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-offset_wq "0" "Offset for the kinematic moisture flux [g kg-1 m s-1]."
  :type 'string
  :group 'mxlch)

(defcustom mxlch-starttime_chem "sunrise"
  "Time after which the chemical emissions start in the case of
  functions 2 and 3. [s]

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist."
  :type 'string
  :group 'mxlch)

(defcustom mxlch-endtime_chem "sunset"
  "Time after which the chemical emissions end in the case of
  functions 2 and 3. [s]

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist."
  :type 'string
  :group 'mxlch)

(defcustom mxlch-starttime_adv "sunrise"
  "Time after which the advection of potential temperature and moisture starts [s].

Not sure about exclamation syntax, but the true variable name
does not have this (are these just comments in the namelist?).

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-endtime_adv "sunset"
  "Time after which the advection of potential temperature and moisture ends [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist."
  :type 'string
  :group 'mxlch)


(defcustom mxlch-function_wt "2"
  "Shape of the kinematic heat flux.

0 = No flux

1 = Constant flux

2 = Sinusoid flux evolution with a start and an end time.

3 = Constant flux with a start and end time.

4 = Cosine shaped flux with a start and an end time. Equal to 0
at start and end and to `mxlch-wthetasmax' in the
middle (Standard cosine is multiplied by -`mxlch-wthetasmax'/2
and shifted by `mxlch-wthetasmax'/2)."
  :type 'string
  :group 'mxlch)

(defcustom mxlch-function_wq "2"
  "Shape of the kinematic moisture flux (see `mxlch-function_wt')."
  :type 'string
  :group 'mxlch)

;; NAMSOA
;; all of below are undocumented, and have no defaults excelt lvbs.
(defcustom mxlch-lvbs ".false."
  "Undocumented. This defaults to .false. in fortran files. If it
  is true, the namelist must define all other variables in
  NAMSOA (there are no defaults)."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-low_high_NOx "1"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha1_TERP_low "0.107"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha2_TERP_low "0.092"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha3_TERP_low "0.359"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha4_TERP_low "0.600"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha1_TERP_high "0.012"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha2_TERP_high "0.122"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha3_TERP_high "0.201"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha4_TERP_high "0.500"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha1_ISO_low "0.009"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha2_ISO_low "0.030"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha3_ISO_low "0.015"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha1_ISO_high "0.001"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlnch-alpha2_ISO_high "0.023"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)
(defcustom mxlch-alpha3_ISO_high "0.015"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt."
  :type 'string
  :group 'mxlch)


;;;; functions for setting up and running my experiments
(defun mxlch-set-northwoods-non-default-constants ()
  "Set all mxlch variables that deviate from the defaults, for the northwoods in central park."
  (setq mxlch-dtime "60")
  ;; coordinates of northwoods in central park
  (setq mxlch-latt "40.797327")
  ;; (+ 360.0 -73.956050) 286.04395
  (setq mxlch-long "286.04395")
  ;; for now set hour to 0 rather than sunrse, to give initial
  ;; conditions time to "spin up". note his is not a modifcation from
  ;; default
  (setq mxlch-hour "0.0")
  (setq mxlch-lenhancedentrainment ".true.")
  (setq mxlch-lradiation ".true.")
  (setq mxlch-llandsurface ".true.")
  (setq mxlch-lrsAgs ".true.")
  (setq mxlch-lCO2Ags ".true.")
  (setq mxlch-c_ustr ".false.")
  (setq mxlch-lscu ".true.")

  ;; do we want to prescribe no fluxes, or a constant prescribed flux set to *max?
  (setq mxlch-function_wt "0")
  (setq mxlch-function_wq "0")

  ;; below land parameters from broadleaf forest config in CLASS (see landsoil.cpp)
  (setq mxlch-z0 "2.0")
  (setq mxlch-z0m "2.0")
  (setq mxlch-z0h "2.0")
  (setq mxlch-albedo "0.25" )
  (setq mxlch-LAI "5" )
  (setq mxlch-Lambda "20.0" )

  ;; should not be used if we use lrsAgs, but setting just in case
  (setq mxlch-gD "0.03" )
  (setq mxlch-rsmin "200.0")
  't)

(defun mxlch-set-non-default-constants ()
  "Set all mxlch variables that deviate from the defaults, but are still held constant across experiments."
  ;; do we want to prescribe no fluxes, or a constant prescribed flux set to *max?
  (setq mxlch-function_wt "0")
  (setq mxlch-function_wq "0")
  (setq mxlch-atime "300")
  (setq mxlch-atime_vert "86400")
  't)




(provide 'mxlch)
;;; mxlch.el ends here
