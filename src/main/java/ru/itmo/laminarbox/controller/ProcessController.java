package ru.itmo.laminarbox.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/process")
public class ProcessController {

    @GetMapping
    public String getPage() {
        return "process";
    }

}
