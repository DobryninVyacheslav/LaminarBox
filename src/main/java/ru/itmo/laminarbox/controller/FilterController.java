package ru.itmo.laminarbox.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
@RequestMapping("/filter")
public class FilterController {

    @GetMapping
    public String getPage() {
        return "filter";
    }

}
