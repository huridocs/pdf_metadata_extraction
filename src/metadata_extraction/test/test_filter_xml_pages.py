from unittest import TestCase

from metadata_extraction.FilterValidSegmentsPages import FilterValidSegmentPages


class TestFilterXmlPages(TestCase):
    def test_filter_xml_pages(self):
        xml_text = """<foo />
        <Page ID="Page1" PHYSICAL_IMG_NR="1" >
            <bar>
        </Page><Page ID="Page2" PHYSICAL_IMG_NR="2" >
            <bar>
        </Page><Page ID="Page3" PHYSICAL_IMG_NR="3" >
            <bar>
        </Page>
        <foo />"""

        xml_expect = """<foo />
        <Page ID="Page2" PHYSICAL_IMG_NR="2" >
            <bar>
        </Page>
        <foo />"""

        self.assertEqual(xml_expect.split(), FilterValidSegmentPages.filter_xml_pages(xml_text, [2]).split())

    def test_filter_xml_pages_one_line(self):
        xml_text = """<foo /><Page ID="Page1" PHYSICAL_IMG_NR="1" >
            <bar>
        </Page>
        <Page ID="Page2" PHYSICAL_IMG_NR="2" >
            <bar>
        </Page>
        <Page ID="Page3" PHYSICAL_IMG_NR="3" >
            <bar>
        </Page>
        <foo />"""

        xml_text = xml_text.replace('\n', ' ')
        xml_expect = """<foo />
        <Page ID="Page2" PHYSICAL_IMG_NR="2" >
            <bar>
        </Page>
        <foo />"""

        self.assertEqual(xml_expect.split(), FilterValidSegmentPages.filter_xml_pages(xml_text, [2]).split())

    def test_filter_xml_pages_different_ids_order(self):
        xml_text = """<foo />
        <Page ID="Page1" PHYSICAL_IMG_NR="1" >
            <bar>
        </Page>
        <Page PHYSICAL_IMG_NR="2" ID="Page2"  >
            <bar>
        </Page>
        <Page PHYSICAL_IMG_NR="3" ID="Page3"  >
            <bar>
        </Page>
        <foo />"""

        xml_expect = """<foo />
        <Page PHYSICAL_IMG_NR="2" ID="Page2" >
            <bar>
        </Page>
        <foo />"""

        self.assertEqual(xml_expect.split(), FilterValidSegmentPages.filter_xml_pages(xml_text, [2]).split())

    def test_filter_xml_pages_first_pages(self):
        xml_text = """<foo />
        <Page ID="Page1" PHYSICAL_IMG_NR="1" >
            <bar>
        </Page>
        <Page PHYSICAL_IMG_NR="2" ID="Page2"  >
            <bar>
        </Page>
        <Page PHYSICAL_IMG_NR="3" ID="Page3"  >
            <bar>
        </Page>"""

        xml_expect = """<foo />
        <Page ID="Page1" PHYSICAL_IMG_NR="1" >
            <bar>
        </Page>
        <Page PHYSICAL_IMG_NR="2" ID="Page2"  >
            <bar>
        </Page>"""

        self.assertEqual(xml_expect.split(), FilterValidSegmentPages.filter_xml_pages(xml_text, [1, 2]).split())

    def test_filter_xml_pages_last_page(self):
        xml_text = """<foo />
        <Page ID="Page1" PHYSICAL_IMG_NR="1" >
            <bar>
        </Page>
        <Page PHYSICAL_IMG_NR="2" ID="Page2"  >
            <bar>
        </Page>
        <Page PHYSICAL_IMG_NR="3" ID="Page3"  >
            <bar>
        </Page>"""

        xml_expect = """<foo />
        <Page PHYSICAL_IMG_NR="3" ID="Page3"  >
            <bar>
        </Page>"""

        self.assertEqual(xml_expect.split(), FilterValidSegmentPages.filter_xml_pages(xml_text, [3]).split())

    def test_filter_xml_pages_id_with_zeros(self):
        xml_text = """<foo />
        <Page ID="Page00001" PHYSICAL_IMG_NR="01" >
            <bar>
        </Page>
        <Page ID="Page0002" PHYSICAL_IMG_NR="002" >
            <bar>
        </Page>
        <Page ID="Page003" PHYSICAL_IMG_NR="0003" >
            <bar>
        </Page>
        <foo />"""

        xml_expect = """<foo />
        <Page ID="Page0002" PHYSICAL_IMG_NR="002" >
            <bar>
        </Page>
        <foo />"""

        self.assertEqual(xml_expect.split(), FilterValidSegmentPages.filter_xml_pages(xml_text, [2]).split())

    def test_filter_xml_should_do_nothing_when_not_matching_tags(self):
        xml_text_more_opening_ones = """<foo />
        <Page ID="Page00001" PHYSICAL_IMG_NR="01" >
            <bar>
        </Page>
        <Page ID="Page0002" PHYSICAL_IMG_NR="002" >
            <Page ID="Page1000" PHYSICAL_IMG_NR="1000"
            <bar>
        </Page>
        <Page ID="Page003" PHYSICAL_IMG_NR="0003" >
            <bar>
        </Page>
        <foo />"""

        xml_text_more_opening_ones_2 = """<foo />
        <Page ID="Page00001" PHYSICAL_IMG_NR="01" >
            <bar>
        </Page>
        <Page ID="Page0002" PHYSICAL_IMG_NR="002" >
            PHYSICAL_IMG_NR="2000"
            <bar>
        </Page>
        <Page ID="Page003" PHYSICAL_IMG_NR="0003" >
            <bar>
        </Page>
        <foo />"""

        xml_text_more_closing = """<foo />
        <Page ID="Page00001" PHYSICAL_IMG_NR="01" >
            <bar>
        </Page>
        <Page ID="Page0002" PHYSICAL_IMG_NR="002" >
            <bar>
                    </Page>
        </Page>
        <Page ID="Page003" PHYSICAL_IMG_NR="0003" >
            <bar>
        </Page>
        <foo />"""

        self.assertEqual(
            xml_text_more_opening_ones.split(),
            FilterValidSegmentPages.filter_xml_pages(xml_text_more_opening_ones, [2]).split(),
        )

        self.assertEqual(
            xml_text_more_opening_ones_2.split(),
            FilterValidSegmentPages.filter_xml_pages(xml_text_more_opening_ones_2, [2]).split(),
        )

        self.assertEqual(
            xml_text_more_closing.split(), FilterValidSegmentPages.filter_xml_pages(xml_text_more_closing, [2]).split()
        )