# -*- coding: utf-8 -*-
import scrapy


class Valscraper1Spider(scrapy.Spider):
    name = 'valscraper1'
    start_urls = ['https://www.amherst.edu/campuslife/housing-dining/dining/menu/2014-08-31/2014-09-07']

    def parse(self, response):
		meals = response.xpath('//div[@class="dining-menu-meal"]')
		for meal in meals:
			date = meal.xpath('a/@data-date').extract_first()
			meal_type = meal.xpath('a/@data-meal').extract_first()

			categories = meal.xpath('div/div[@class="dining-course-name"]/text()').extract()
			contents = meal.xpath('div/p/text()').extract()
			main_meal = ''
			if len(contents) > 0:
				main_meal = contents[-1]

			full_meal = {}
			for i in range(0,len(categories)):
				full_meal[categories[i]] = contents[i]

			yield{
				'date' : date,
				'type' : meal_type,
				'main' : main_meal,
				'full' : full_meal
			}

		next_page = response.xpath('//div[@id="dining-hall-next-week"]').css('a::attr(href)').extract_first()
		if next_page is not None:
			next_page = response.urljoin(next_page)
			yield scrapy.Request(next_page, callback = self.parse)

