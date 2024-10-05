F1::
{
loop 15
{
WinActivate ("ahk_id 4528874")
Send "^r"
Sleep 7500
MouseClick "left", 2178, 429
MouseClick "left", 2178, 429
Send "^c"
Sleep 50
WinActivate ("ahk_exe WindowsTerminal.exe")
SendText "py tiktokscrape.py"
Sleep 50
Send "{Enter}"
Sleep 3000
Send "^v"
Sleep 50
Send "{Enter}"
Sleep 240000
}
}

F2::ExitApp