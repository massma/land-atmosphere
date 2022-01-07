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

(defvar mxlch-dir nil
  "The director to find the MXLCH repository. See https://github.com/classmodel/mxlch.")


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
(defvar mxlch-wthetasmax "0.0"
  "Maximum surface kinematic heat flux for standard flux profiles in K m s^{-1}.")
(defvar mxlch-c_fluxes ".false."
  "If true all fluxes are constant. It is better to use the options in NAMFLUX.")
(defvar mxlch-gamma "0.006"
  "Potential temperature lapse rate in the free troposphere in K m^{-1}.")
(defvar mxlch-thetam0 "295.0"
  "Initial mixed layer potential temperature in K.")
(defvar mxlch-dtheta0 "4"
  "Initital potential temperature jump in K.")
(defvar mxlch-advtheta "0.0"
  "Advection of potential temperature in K s^{-1}.")
(defvar mxlch-pressure "1013.0"
  "Air pressure in the boundary layer in Pa.")
(defvar mxlch-wqsmax "0.00"
  "Maximum surface kinematic moisture flux for standard flux profiles, in g kg^{-1} m s^{-1}.")
(defvar mxlch-gammaq "0.0"
  "Specific humidity lapse rate in the free troposphere in g kg^{-1} m^{-1}.")
(defvar mxlch-qm0 "0"
  "Initial mixed layer specific humidity in g kg^{-1}.")
(defvar mxlch-dq0 "0"
  "Inititial specific humidity jump in g kg^{-1}.

Must be greater than or equal to  -`mxlch-qm0'.")
(defvar mxlch-advq "0.0"
  "Advection of specific humidity. g kg^{-1} s^{-1}.")
(defvar mxlch-wcsmax "0.0"
  "Maximum surface kinematic tracer flux for standard flub proiles in ppb m s^{-1}.")
(defvar mxlch-gammac "0.0"
  "Tracer lapse rate in the free troposphere in ppb m^{-1}.")
(defvar mxlch-cm0 "0.0"
  "Initital mixed layer tracer concentration in ppb.")
(defvar mxlch-dc0 "0.0"
  "Initial tracer concentration part in ppb.

Must be greater than or equal to -`mxlch-cm0'.")
(defvar mxlch-c_ustr ".false."
  "If true the momentum fluxes (and friction velocity) are constant.")
(defvar mxlch-z0 "0.03"
  "Roughness length in m.")
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
(defvar mxlch-lencroachment ".false." "Enables encroachment")
(defvar mxlch-lscu ".false." "Enables shallow cumulus mass-flux parameterization.")
(defvar mxlch-lrelaxdz ".false." "If true, LCL-z_i is nudged to calculated value by time-scale \tau, rather than being set to that value.")
(defvar mxlch-tau "7200" "Time-scale for nudging transition layer depth in seconds.")
(defvar mxlch-ladvecFT ".false." "TODO: where is doc??")

;; NAMSURFLAYER
(defvar mxlch-lsurfacelayer ".false." "Enable or disable surface layer.")
(defvar mxlch-z0m "0.03" "Roughness length of momentum in m.")
(defvar mxlch-z0h "0.03" "Roughness lenth of heat in m.")

;; NAMRAD
(defvar mxlch-lradiation ".false." "Enable or disable radiation.")
(defvar mxlch-cc "0" "Amount of cloud cover; unitless between 0 and 1.")
(defvar mxlch-S0 "1368" "Incoming shortwave solar radiation in W m^{-2}.")
(defvar mxlch-albedo "0.2" "Albedo as a fraction between 0 and 1.")

;; NAMSURFACE
(defvar mxlch-llandsurface "switch to use interactive landsurface")
(defvar mxlch-Qtot "Incoming energy")
(defvar mxlch-lsea "Using a sea surface instead of land")
(defvar mxlch-sst "Sea surface temperature")
(defvar mxlch-Ts "Initial surface temperature [K]")
(defvar mxlch-wwilt "wilting point")
(defvar mxlch-w2 "Volumetric water content deeper soil layer")
(defvar mxlch-wg "Volumetric water content top soil layer")
(defvar mxlch-wfc "Volumetric water content field capacity")
(defvar mxlch-wsat "Saturated volumetric water content ECMWF config")
(defvar mxlch-CLa "Clapp and Hornberger retention curve parameter a")
(defvar mxlch-CLb "Clapp and Hornberger retention curve parameter b")
(defvar mxlch-CLc "Clapp and Hornberger retention curve parameter c")
(defvar mxlch-C1sat "Coefficient force term moisture")
(defvar mxlch-C2ref "Coefficient restore term moisture")
(defvar mxlch-gD "VPD correction factor for rs")
(defvar mxlch-rsmin "Minimum resistance of transpiration")
(defvar mxlch-rssoilmin "Minimum resistance of soiltranspiration")
(defvar mxlch-LAI "Leaf area index")
(defvar mxlch-cveg "Vegetation fraction")
(defvar mxlch-Tsoil "Temperature top soil layer")
(defvar mxlch-T2 "Temperature deeper soil layer")
(defvar mxlch-Wl "Equivalent water layer depth for wet vegetation")
(defvar mxlch-Lambda "Thermal diffusivity skin layer")
(defvar mxlch-CGsat "Saturated soil conductivity for heat")
(defvar mxlch-lrsAgs "Switch to use A-gs model for surface resistances")
(defvar mxlch-lCO2Ags "Switch to use A-gs model for CO2 flux")
(defvar mxlch-CO2comp298 "CO2 compensation concentration [mg m-3]")
(defvar mxlch-Q10CO2 "function parameter to calculate CO2 compensation concentration [-]")
(defvar mxlch-gm298 "mesophyill conductance at 298 K [mm s-1]")
(defvar mxlch-Ammax298 "CO2 maximal primary productivity [mg m-2 s-1]")
(defvar mxlch-Q10gm "function parameter to calculate mesophyll conductance [-] ")
(defvar mxlch-T1gm "reference temperature to calculate mesophyll conductance gm [K]")
(defvar mxlch-T2gm "reference temperature to calculate mesophyll conductance gm [K]")
(defvar mxlch-Q10Am "function parameter to calculate maximal primary profuctivity Ammax")
(defvar mxlch-T1Am "reference temperature to calculate maximal primary profuctivity Ammax [K] ")
(defvar mxlch-T2Am "reference temperature to calculate maximal primary profuctivity Ammax [K]")
(defvar mxlch-f0 "maximum value Cfrac [-]")
(defvar mxlch-ad "regression coefficient to calculate Cfrac [kPa-1]")
(defvar mxlch-alpha0 "initial low light conditions [mg J-1]")
(defvar mxlch-Kx "extinction coefficient PAR [-]")
(defvar mxlch-gmin "cuticular (minimum) conductance [m s-1]")
(defvar mxlch-Cw "constant water stress correction (eq. 13 Jacobs et al. 2007) [-]")
(defvar mxlch-wsmax "upper reference value soil water [-]")
(defvar mxlch-wsmin "lower reference value soil water [-]")
(defvar mxlch-R10 "respiration at 10 C [mg CO2 m-2 s-1]")
(defvar mxlch-Eact0 "activation energy [53.3 kJ kmol-1]")
(defvar mxlch-lBVOC "Enable the calculation of BVOC (isoprene, terpene) emissions")
(defvar mxlch-BaserateIso "Base emission rate for isoprene emissions [microg m^4 h^-1]")
(defvar mxlch-BaserateTer "Base emission rate for terprene emissions [microg m^4 h^-1]")

;; NAMCHEM
(defvar mxlch-lchem ".true.")
(defvar mxlch-lcomplex ".false.")
(defvar mxlch-lwritepl ".true.")
(defvar mxlch-ldiuvar ".true.")
(defvar mxlch-h_ref "12.")
(defvar mxlch-lflux ".false.")
(defvar mxlch-fluxstart "0.0")
(defvar mxlch-fluxend "0.0")
(defvar mxlch-lchconst ".false.")
(defvar mxlch-t_ref_cbl "298.")
(defvar mxlch-p_ref_cbl "1000")
(defvar mxlch-q_ref_cbl "10.0")
(defvar mxlch-t_ref_ft "298.")
(defvar mxlch-p_ref_ft "1000.")
(defvar mxlch-q_ref_ft "10.0")
(defvar mxlch-!pressure_ft "1013.0")

;; NAMFLUX
(defvar mxlch-offset_wt "0")
(defvar mxlch-offset_wq "0")
(defvar mxlch-function_wt "2")
(defvar mxlch-function_wq "2")
(defvar mxlch-starttime_wt "0")
(defvar mxlch-endtime_wt "39600")
(defvar mxlch-starttime_wq "0")
(defvar mxlch-endtime_wq "39600")
(defvar mxlch-starttime_chem "0")
(defvar mxlch-endtime_chem "39600")
(defvar mxlch-!starttime_adv "sunrise")
(defvar mxlch-!endtime_adv "sunset")

;; NAMSOA
(defvar mxlch-lvbs ".true.")
(defvar mxlch-low_high_NOx "1")
(defvar mxlch-alpha1_TERP_low "0.107")
(defvar mxlch-alpha2_TERP_low "0.092")
(defvar mxlch-alpha3_TERP_low "0.359")
(defvar mxlch-alpha4_TERP_low "0.600")
(defvar mxlch-alpha1_TERP_high "0.012")
(defvar mxlch-alpha2_TERP_high "0.122")
(defvar mxlch-alpha3_TERP_high "0.201")
(defvar mxlch-alpha4_TERP_high "0.500")
(defvar mxlch-alpha1_ISO_low "0.009")
(defvar mxlch-alpha2_ISO_low "0.030")
(defvar mxlch-alpha3_ISO_low "0.015")
(defvar mxlch-alpha1_ISO_high "0.001")
(defvar mxlch-alpha2_ISO_high "0.023")
(defvar mxlch-alpha3_ISO_high "0.015")



(provide 'mxlch)
;;; mxlch.el ends here
