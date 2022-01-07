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


;;;;  name options

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


(provide 'mxlch)
;;; mxlch.el ends here
