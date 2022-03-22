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

(defcustom mxlch-data-dir (concat (getenv "HOME") "/land-atmosphere/data")
  "The directory to find MXLCH data and simulations, as output by soundings.py."
  :type 'string
  :group 'mxlch)


;;;;  Utility functions
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
  ;; we migth not want this to be interactive: instead call this from the design of my simulations
  (interactive)
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

(defun mxlch-write-namelists ()
  "Write all namelists for experiments in `mxlch-data-dir'."
  (interactive)
  (mxlch-set-non-default-constants)
  (let ((inputs (directory-files-recursively mxlch-data-dir (rx  (*? anything) "input.el"))))
    (dolist (input inputs)
      (load input)
      (mxlch-write-namelist (concat (file-name-directory input)
                                    "namoptions")))
    't))

(defun mxlch-run-models ()
  "Run all models for experiments in `mxlch-data-dir'.

This could be asnychronous, but we end up with too many pipes open.

Would have to use a more sophisticated handler using sentinels in that case."
  (interactive)
  (let ((inputs (directory-files-recursively mxlch-data-dir (rx  (*? anything) "input.el"))))
    (dolist (input inputs)
      (let ((default-directory  (file-name-directory input)))
        (call-process "rm" nil nil nil "-rf" (concat default-directory "RUN00") "model-run.out")
        (call-process
         (concat mxlch-dir "/MXLCH")
         nil
         `(:file ,(concat default-directory "/model-run.out"))
         nil
         (concat mxlch-dir "/chem.inp.op3"))))
    't))

(defun mxlch-extract-et (dir)
  "Extract et from output_land file in experiment DIR.

Currently, this takes all data between 10 (inclusive) and 14 (exclusive)
local time, and averages it.

It returns ET as a string, and if it encounters any NaN's,
it will return \"NaN\"."
  (find-file (concat dir "/RUN00/output_land"))
  (goto-char (point-min))
  (forward-line 3)
  (move-beginning-of-line 1)
  (let ((time)
        (count 0)
        (sum 0.0))
    (while (and
            (re-search-forward (rx (* blank) (group (+ (or num ?.))))
                               (line-end-position)
                               t)
            (numberp sum))
      (setq time (read (match-string 1)))
      (when (and (>= time 10.0)
                 (< time 14.0))
        (dotimes (_x 8)
          (re-search-forward (rx (* blank) (group (or "NaN" (+ (or num ?.)))))
                              (line-end-position)
                              t))
        (let ((num (read (match-string 1))))
          (if (eq 'NaN num)
              (setq sum 'NaN)
            (setq count (+ count 1))
            (setq sum (+ sum num)))))
      (move-beginning-of-line 2))
    (kill-buffer)
    (cond
     ((eq 'NaN sum) "nan")
     ((= 0 count) "nan")
     (t (number-to-string (/ sum (float count)))))))

(defun mxlch-test-extract-et ()
  "Test the function `mxlch-extract-et'."
  (if
      (string= "nan"
          (mxlch-extract-et
           (file-name-directory
            "/home/adam/land-atmosphere/data/reality/kelowna_71203_2010_098/input.el")))
      'ok
    (error
     "FAILED TO PASS TEST ON:
 /home/adam/land-atmosphere/data/reality/kelowna_71203_2010_098/input.el"))
  (if (string= "274.6017542083333"
         (mxlch-extract-et
          (file-name-directory
           "/home/adam/land-atmosphere/data/causal/kelowna_71203_000019/input.el")))
      'ok
    (error
     "FAILED TO PASS TEST ON:
/home/adam/land-atmosphere/data/causal/kelowna_71203_000019/input.el")))

(defun mxlch-run-tests ()
  "Run every test in the mxlch package."
  (mxlch-test-extract-et))

(defun mxlch-load-output (input)
  "Load model output corresponding to input file INPUT.

The returned data structure will be a list of length 3:
(experiment name [string], soil moisture [string], ET [string]."
  (load-file input)
  (let* ((dir (file-name-directory input))
         (et (mxlch-extract-et dir))
         (experiment-name (file-name-base (directory-file-name dir))))
    (list experiment-name mxlch-wg et)))


(defun mxlch-write-csv (dir)
  "Write a csv from all experiments in DIR."
  (let ((filename (concat (directory-file-name dir) ".csv"))
        (inputs (directory-files-recursively dir (rx  (*? anything) "input.el"))))
    (find-file filename)
    (erase-buffer)
    (insert "experiment,SM,ET
")
    (dolist (input inputs)
      (let ((values (mxlch-load-output input)))
        (insert (format "%s,%s,%s
" (nth 0 values) (nth 1 values) (nth 2 values)))))
    (save-buffer)
    (kill-buffer)))

(defun mxlch-write-all-csvs ()
  "Write all model output csvs.

Call `mxlch-write-csv' on every directory in `mxlch-data-dir'."
  (interactive)
  (let ((exp-dirs (directory-files mxlch-data-dir t (rx (not ?.) (*? anything)))))
    (dolist (dir exp-dirs)
      (when (file-directory-p dir)
          (mxlch-write-csv dir)))))



;;;; Name options
;; NAMRUN
(defvar mxlch-outdir "'RUN00'" "Name of output folder. Must be enclosed in \"'\"")
(defvar mxlch-time "86400" "Simulated time in seconds.")
(defvar mxlch-dtime "1" "Timestep in seconds.")
(defvar mxlch-atime "60" "Time interval for statistics in seconds.")
(defvar mxlch-atime_vert "1800" "Time interval for vertical profile statistics in seconds.")
(defvar mxlch-h_max "3000" "Maximum height of the simulated domain in meters.")
(defvar mxlch-latt "0" "Latitude of simulated location in degrees.
Should be between -90 and 90 degrees, inclusive.")
(defvar mxlch-long "0" "Longitude of simulated location in degrees.
Must be between 0 and 360 degrees, inclusive.")
(defvar mxlch-day "80" "Julian day of the year.
Between 1 and 365.")
(defvar mxlch-hour "0" "Local time at which the simulation starts in hours.
Between 0 and 24 hours.")

;; NAMDYN
(defvar mxlch-zi0 "200.0" "Initital boundary layer height in meters.")
(defvar mxlch-beta "0.20" "Entrainment ratio.")
(defvar mxlch-lenhancedentrainment ".false."
  "If true, the entrainment is enhanced by wind shear (u_{star}).")
(defvar mxlch-wsls "0.000" "Flow divergence factor for subsidence, in s^{-1}.")
(defvar mxlch-lfixedlapserates ".false."
  "If true, the (enhancing) effect of subsidence on free tropospheric gradients is omitted.")
(defvar mxlch-lfixedtroposphere ".false."
  "UNDOCUMENTED in pdf.")
(defvar mxlch-wthetasmax "0.0"
  "Maximum surface kinematic heat flux for standard flux profiles in K m s^{-1}.")
(defvar mxlch-c_fluxes ".false."
  "If true all fluxes are constant. It is better to use the options in NAMFLUX.

(also note that this replace c_wth option in the NAMELIST.")
(defvar mxlch-gamma "0.006"
  "Potential temperature lapse rate in the free troposphere in K m^{-1}.")
(defvar mxlch-lgamma ".false."
  "A switch to apply a different lapse rate (`mxlch-gamma2') to heights above some hcrit.

TODO: undocumented in PDF; I could add.")
(defvar mxlch-hcrit "10000"
  "When `mxlch-lgamma' is true, apply `mxlch-gamma2' when zi > `mxlch-hcrit'.

TODO: undocumented in PDF: I could add.")
(defvar mxlch-gamma2 "0.006"
  "When `mxlch-lgamma' is true, apply this lapse rate when zi > hcrit.

TODO: undocumented in PDF: I could add.")
(defvar mxlch-thetam0 "295.0"
  "Initial mixed layer potential temperature in K.")
(defvar mxlch-dtheta0 "4"
  "Initital potential temperature jump in K.")
(defvar mxlch-pressure "1013.0"
  "Air pressure in the boundary layer in hPa. TODO:fix pdf documentation with wrong units!")
(defvar mxlch-wqsmax "0.00"
  "Maximum surface kinematic moisture flux for standard flux profiles, in g kg^{-1} m s^{-1}.")
(defvar mxlch-gammaq "0.0"
  "Specific humidity lapse rate in the free troposphere in g kg^{-1} m^{-1}.")
(defvar mxlch-qm0 "0"
  "Initial mixed layer specific humidity in g kg^{-1}.")
(defvar mxlch-dq0 "0"
  "Inititial specific humidity jump in g kg^{-1}.

Must be greater than or equal to  -`mxlch-qm0'.")
(defvar mxlch-wcsmax "0.0"
  "Maximum surface kinematic tracer flux for standard flub proiles in ppb m s^{-1}.")
(defvar mxlch-gammac "0.0"
  "Tracer lapse rate in the free troposphere in ppb m^{-1}.")
(defvar mxlch-cm0 "0.0"
  "Initital mixed layer tracer concentration in ppb.
\(confirmed, I think, in code that this is in fact ppb and NOT ppm)

I think the model interprets this as carbon dioxide; see note in .f90:

\"! cm0: initial carbon dioxide mixing layer value
! dc0: initial carbon dioxide jump
\"")
(defvar mxlch-dc0 "0.0"
  "Initial tracer concentration part in ppb.
\(confirmed, I think, in code that this is in fact ppb and NOT ppm)

Must be greater than or equal to -`mxlch-cm0'

I think the model interprets this as carbon dioxide.

See note in `mxlch-cm0' docs.")
(defvar mxlch-c_ustr ".true."
  "If true the momentum fluxes (and friction velocity) are constant.

TODO: change in code to be lc_ustr to match coding convention on logical variables?

This would make a backwards incompatable change to the namelist though, which
might not be ideal.")
(defvar mxlch-z0 "0.03"
  "Roughness length in m.

TODO: Remove misleading comment in the fortran files falsly
stating that z0 is \"initial boundary layer height\".")

(defvar mxlch-uws0 "0"
  "Initital surface (x-)momentum flux in m^2 s^{-2}.")
(defvar mxlch-vws0 "0"
  "Initial surface (y-)momentum flux in m^2 s^{-2}.")
(defvar mxlch-gammau "0"
  "Lapse rate of u in the free troposphere in s^{-1}.")
(defvar mxlch-gammav "0"
  "Lapse rate of of v in the free troposphere in s ^{-1}.")
(defvar mxlch-um0 "0"
  "Initital u in the mixed layer in m s^{-1}.")
(defvar mxlch-vm0 "0"
  "Initital v in the mixed layer in m s^{-1}.")
(defvar mxlch-ug "0"
  "Geostrophic wind in the x-direction in m s^{-1}.")
(defvar mxlch-vg "0"
  "Geostrophic wind in the y-direction in m s^{-1}.")
(defvar mxlch-advq "0.0"
  "Advection of specific humidity. g kg^{-1} s^{-1}.")
(defvar mxlch-advtheta "0.0"
  "Advection of potential temperature in K s^{-1}.")
(defvar mxlch-ladvecFT ".false." "If true advection is also applied for free troposphere.")
(defvar mxlch-lencroachment ".false." "Enables encroachment")
(defvar mxlch-lscu ".false." "Enables shallow cumulus mass-flux parameterization.")
(defvar mxlch-lrelaxdz ".false." "If true, LCL-z_i is nudged to calculated value by time-scale \tau, rather than being set to that value.")
(defvar mxlch-tau "7200" "Time-scale for nudging transition layer depth in seconds.")


;; NAMSURFLAYER
(defvar mxlch-lsurfacelayer ".false." "Enable or disable surface layer.")
(defvar mxlch-z0m "0.03" "Roughness length of momentum in m.

Can get good defaults from the landsoil.cpp file in CLASS.")
(defvar mxlch-z0h "0.03" "Roughness lenth of heat in m.

Can get good defaults from the landsoil.cpp file in CLASS.")

;; NAMRAD
(defvar mxlch-lradiation ".false." "Enable or disable
radiation. If this is turned off, net surface radiation is just a
constant.")
(defvar mxlch-cc "0" "Amount of cloud cover; unitless between 0 and 1.")
(defvar mxlch-S0 "1368" "Incoming shortwave solar radiation in W m^{-2}.")
(defvar mxlch-DeltaFsw "0.0" "Absorbed radiation by e.g. aerosols (neg. value) (shortwave component). [UNDOCUMENTED]")
(defvar mxlch-DeltaFlw "0.0" "Emitted radiation by e.g. clouds (pos. value) (longwave compoonent). [UNDOCUMENTED]")
(defvar mxlch-Rdistr "1.0" "Distribution of absorbing aerosols (see Barbero et al., 2013)")
(defvar mxlch-albedo "0.2" "Albedo as a fraction between 0 and 1.")

;; NAMSURFACE
(defvar mxlch-llandsurface ".false." "switch to use interactive landsurface")
(defvar mxlch-Qtot "400" "Incoming energy in W m^{-2}")
(defvar mxlch-lsea ".false." "Using a sea surface instead of land.")
(defvar mxlch-sst "285" "Sea surface temperature in Kelvin.")
(defvar mxlch-Ts "295.0"
  "Initial surface temperature [K].

Note in MXLCH this defaults to `mxlch-thetam0', however in the
.el interface it will always be set to this value.")
(defvar mxlch-wwilt "0.314" "Wilting point (m^3 m^{-3}).")
(defvar mxlch-w2 "0.420" "Volumetric water content deeper soil layer. (m^3 m^{-3})")
(defvar mxlch-wg "0.40" "Volumetric water content top soil layer (m^3 m^{-3}).")
(defvar mxlch-wfc "0.491" "Volumetric water content field capacity (m^3 m^{-3}).")
(defvar mxlch-wsat "0.6" "Saturated volumetric water content ECMWF config (m^3 m^{-3}).")
(defvar mxlch-CLa "0.083" "Clapp and Hornberger retention curve parameter a [unitless].")
(defvar mxlch-CLb "11.4" "Clapp and Hornberger retention curve parameter b [unitless].")
(defvar mxlch-CLc "12.0" "Clapp and Hornberger retention curve
parameter c [unitless] (this is parameter \"p\" in textbook, soo
page 140).")
(defvar mxlch-C1sat "0.342" "Coefficient force term moisture (for heat, see page 140 in textbook).")
(defvar mxlch-C2ref "0.3" "Coefficient restore term moisture (for heat, see page 140 in textbook).")
(defvar mxlch-gD "0.0" "VPD correction factor for rs TODO:units?. This is only used of lrsAgs is false. Also the formulation for rs doesn't make sense to me, something to check.")
(defvar mxlch-rsmin "0.0" "Minimum resistance of transpiration (s m^{-1}).

This is only used of lrsAgs is false. Also the formulation for rs
doesn't make sense to me, something to check. Basically if rsmin
is 0.0, then rs will be 0.0 independent of all the other factors
and vpd, etc.")
(defvar mxlch-rssoilmin "0.0" "Minimum resistance of soiltranspiration [s m^{-1}].

It seems like this formulation is broken: if rssoilmin is 0.0, then it will always be 0.0 independent of the other factors. look up Jarvis-Steward stomatal conudctance model.")
(defvar mxlch-LAI "1.0" "Leaf area index [m^2 m^{-2}].")
(defvar mxlch-cveg "1.0" "Vegetation fraction")
(defvar mxlch-Tsoil "285" "Temperature top soil layer [K].")
(defvar mxlch-T2 "285" "Temperature deeper soil layer [K]")
(defvar mxlch-Wl "0.0" "Equivalent water layer depth for wet vegetation [m]")
(defvar mxlch-Lambda "5.9" "Thermal diffusivity skin layer [W m^{-2} K^{-1}].")
(defvar mxlch-CGsat "3.6e-6" "Saturated soil conductivity for heat [K m^{2} J^{-1}].")
(defvar mxlch-lrsAgs ".false." "Switch to use A-gs model for surface resistances")
(defvar mxlch-lCO2Ags ".false" "Switch to use A-gs model for CO2 flux")
(defvar mxlch-CO2comp298 "68.5" "CO2 compensation concentration [mg m-3].

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-Q10CO2 "1.5" "function parameter to calculate CO2 compensation concentration [-]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-gm298 "7" "mesophyill conductance at 298 K [mm s-1]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-Ammax298 "2.2" "CO2 maximal primary productivity [mg m-2 s-1]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-Q10gm "2" "function parameter to calculate mesophyll conductance [-]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-T1gm "278" "reference temperature to calculate mesophyll conductance gm [K]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-T2gm "301" "reference temperature to calculate mesophyll conductance gm [K]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-Q10Am "2" "function parameter to calculate maximal primary profuctivity Ammax

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-T1Am "281" "reference temperature to calculate maximal primary profuctivity Ammax [K]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-T2Am "311" "reference temperature to calculate maximal primary profuctivity Ammax [K]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-f0 "0.89" "maximum value Cfrac [-]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-ad "0.07" "regression coefficient to calculate Cfrac [kPa^{-1}]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-alpha0 "0.017" "initial low light conditions [mg J-1]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-Kx "0.7" "extinction coefficient PAR [-]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-gmin "2.5e-4" "cuticular (minimum) conductance [m s-1]

Matches for c3 plants in the CLASS model (see model.cpp).")
(defvar mxlch-Cw "1.6e-3" "constant water stress correction (eq. 13 Jacobs et al. 2007) [-]")
(defvar mxlch-wsmax "0.55" "upper reference value soil water [-]")
(defvar mxlch-wsmin "0.005" "lower reference value soil water [-]")
(defvar mxlch-R10 "0.23" "respiration at 10 C [mg CO2 m-2 s-1]")
(defvar mxlch-Eact0 "53.3e3" "activation energy [53.3 kJ kmol-1]")
(defvar mxlch-lBVOC ".false." "Enable the calculation of BVOC (isoprene, terpene) emissions.

UNDOCUMENTED in pdf.")
(defvar mxlch-BaserateIso "0.0" "Base emission rate for isoprene emissions [microg m^4 h^-1].

UNDOCUMENTED in pdf.")
(defvar mxlch-BaserateTer "0.0" "Base emission rate for terprene emissions [microg m^4 h^-1].

UNDOCUMENTED in pdf.")

;; NAMCHEM
(defvar mxlch-lchem ".false." "Enable or disable chemistry.")
(defvar mxlch-lwritepl ".true." "Enable the output of production and loss terms per chemical.")
(defvar mxlch-lcomplex ".false." "Choice between complex chemical scheme and simplified scheme.")
(defvar mxlch-ldiuvar ".true." "If false the UC radiation during day is calculated at time h.ref.")
(defvar mxlch-h_ref "12." "Reference time for calculated UC radiation if ldiuvar is set to .false.")
(defvar mxlch-lflux ".false." "If set to .true. the times of sunrise and sunset are input. The otpions in NAMFLUX are preferred.")
(defvar mxlch-fluxstart "0.0" "Time of sunrise if lflux is set to .true. [hr].")
(defvar mxlch-fluxend "0.0" "Time of sunset if lflux is set to .true. [hr].")
(defvar mxlch-pressure_ft "1013.0"
  "Defaults to pressure [hPa] in fortran code.

TODO: fix pdf documentation to match units")
(defvar mxlch-lchconst ".false."
  "Switch to calculate reaction rates using reference
  temperatures, humidities, and pressures instead of actual
  values.")
(defvar mxlch-t_ref_cbl "298."
  "Reference temperature in the boundary layer [K].")
(defvar mxlch-p_ref_cbl "1013.5"
  "Reference pressure in the boundary layer [hPa].

TODO: pdf reference has the wrong default (1000.0) and wrong units.!")
(defvar mxlch-q_ref_cbl "10.0"
  "Reference specific humidity in the boundary layer [g kg-1].

")
(defvar mxlch-t_ref_ft "298."
  "Reference temeprature in the free troposphere [K].

")
(defvar mxlch-p_ref_ft "1013.5" "Reference pressure in the free
troposphere [hPa]

I do not think tihs actually gets used, but I could be wrong..

TODO: pdf reference has the wrong default (1000.0) and wrong units.!")
(defvar mxlch-q_ref_ft "10.0"
  "Reference specific humidty in the
free troposphere. [g kg-1].")

;; NAMFLUX
(defvar mxlch-starttime_wt "sunrise"
  "Time after which the heat flux starts in the case of functions 2 and 3 [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist.")
(defvar mxlch-endtime_wt "sunset"
  "Time after which the heat flux ends in the case of functions 2 and 3 [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist.")
(defvar mxlch-offset_wt "0" "Offset for the kinematic heat flux [K m s-1].")

(defvar mxlch-starttime_wq "sunrise"
  "Time after which the moisture flux starts in the case of functions 2 and 3 [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist.")
(defvar mxlch-endtime_wq "sunset"
  "Time after which the moisture flux ends in the case of functions 2 and 3 [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist.")
(defvar mxlch-offset_wq "0" "Offset for the kinematic moisture flux [g kg-1 m s-1].")

(defvar mxlch-starttime_chem "sunrise"
  "Time after which the chemical emissions start in the case of
  functions 2 and 3. [s]

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist.")

(defvar mxlch-endtime_chem "sunset"
  "Time after which the chemical emissions end in the case of
  functions 2 and 3. [s]

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist.")

(defvar mxlch-starttime_adv "sunrise"
  "Time after which the advection of potential temperature and moisture starts [s].

Not sure about exclamation syntax, but the true variable name
does not have this (are these just comments in the namelist?).

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist.")
(defvar mxlch-endtime_adv "sunset"
  "Time after which the advection of potential temperature and moisture ends [s].

Note that currently this is not written to namelist, because the only way to trigger
a sunrise startime is to not have this option in the namelist.")


(defvar mxlch-function_wt "2"
  "Shape of the kinematic heat flux.

0 = No flux

1 = Constant flux

2 = Sinusoid flux evolution with a start and an end time.

3 = Constant flux with a start and end time.

4 = Cosine shaped flux with a start and an end time. Equal to 0
at start and end and to `mxlch-wthetasmax' in the
middle (Standard cosine is multiplied by -`mxlch-wthetasmax'/2
and shifted by `mxlch-wthetasmax'/2).")

(defvar mxlch-function_wq "2"
  "Shape of the kinematic moisture flux (see `mxlch-function_wt').")

;; NAMSOA
;; all of below are undocumented, and have no defaults excelt lvbs.
(defvar mxlch-lvbs ".false."
  "Undocumented. This defaults to .false. in fortran files. If it
  is true, the namelist must define all other variables in
  NAMSOA (there are no defaults).")
(defvar mxlch-low_high_NOx "1"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha1_TERP_low "0.107"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha2_TERP_low "0.092"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha3_TERP_low "0.359"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha4_TERP_low "0.600"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha1_TERP_high "0.012"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha2_TERP_high "0.122"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha3_TERP_high "0.201"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha4_TERP_high "0.500"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha1_ISO_low "0.009"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha2_ISO_low "0.030"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha3_ISO_low "0.015"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha1_ISO_high "0.001"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha2_ISO_high "0.023"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")
(defvar mxlch-alpha3_ISO_high "0.015"
  "Undocumented in pdf, and no defaults in bulk_chemistry.f90, so
  defauls are taken from namoptions.hyyt.")


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
  't)




(provide 'mxlch)
;;; mxlch.el ends here
