#Requires AutoHotkey v2.0

F1::
{
CoordMode "Mouse", "Screen"
loop 1
{
WinActivate ("ahk_exe EXCEL.EXE")
Send "^v"
Send "{down}"
}
}

F2::ExitApp