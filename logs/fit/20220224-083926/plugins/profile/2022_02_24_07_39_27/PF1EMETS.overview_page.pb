?&$	)q???????,`?Up????b????!???~???$	??ҡ??@?3??o???JC?+?
@!*?|??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0wd?6?/??)??????A?<??tZ??YZ/?r?]??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?E???`???U?p??A????_Z??YV?P?????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails08h????? A?c?]??A?Ky ???Y???[?d??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????c??Ps?"???Aō[????Y?Gp#e???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?+j0???Y?b+h??A?0?????Y??7/N|??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Li?-??Hū?m??A?-?熦??Y?%??s|??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???~????Y?H?s??A?Q+L?k??Y5c?tv2??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??C5%Y????ĬC??A&??'d???Y???_?|??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
W\??(??P??0{???ASςP????YP???<??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????Lk??^??Am?%?????Yw1?t????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???_??????-???A?3?%??Y??xy:??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0O;?5Y#??Z?Pۆ??A???[?d??Yv4?????rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?h?^????U??6o??AM?]~??Y????6??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??b????f?2?}???A????	??Y?4?????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?????%???2p@K??A?;?????Yo???I~??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?o???S??Y???tw??A?+f????Y????·??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?t??????4M?~2???A?M+?@.??Y???5"??rtrain 97*	<?O??e?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???t!V??!?3d:??C@)?A?p?-??17T??uB@:Preprocessing2E
Iterator::Root:?!y??!xe?(m>@@)??i?????1X~<[0@:Preprocessing2T
Iterator::Root::ParallelMapV2?|?H?F??!?L?P?!0@)?|?H?F??1?L?P?!0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip@N?0????!D͜k??P@)???????1???cC$@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??;Mf???!Q&\??? @)??;Mf???1Q&\??? @:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMaps??=Ab??!}宇??1@)??
/???1??yQ??@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate???)??!Ӡf?`*@)3k) ???1?Y?L??@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???B????!i????@)???B????1i????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 ?߄&z@I???]?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?މ&???0?x??? A?c?]??!4M?~2???	!       "	!       *	!       2$	????[??n??ʟ?????	??!?Q+L?k??:	!       B	!       J$	??0?r??~m???*d??%??s|??!????6??R	!       Z$	??0?r??~m???*d??%??s|??!????6??b	!       JCPU_ONLYY ?߄&z@b q???]?W@Y      Y@q3]????K@"?	
both?Your program is POTENTIALLY input-bound because 48.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?55.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 