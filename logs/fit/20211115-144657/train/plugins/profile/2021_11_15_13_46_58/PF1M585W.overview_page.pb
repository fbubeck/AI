?($	` ?P??????O?(???,e?X??!??o_??$	+h\+?j@ U?Ƹ?@??mk?@!`??~?3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$f?c]?F???.n????Ax$(~??Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?5^?I??0L?
F%??A_?Q???Y o?ŏ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&mV}??b??0?'???AΈ?????Y?Fx$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?;Nё\????A?f??A??ڊ?e??Y?|гY???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"?uq??]?C?????A~8gDi??Y|??Pk???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*????i o????A?/?$??Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?(\????????B?i??A?MbX9??Y??#?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t??????3???A??H?}??YaTR'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??z6???q=
ףp??AV}??b??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	ۊ?e????2w-!???A(~??k	??Y??Ɯ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??N@a??Q?|a??A=
ףp=??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S??:????A?f??A46<???Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??N@a??6?;Nё??A??ׁsF??Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,Ԛ??????6?[??A??<,Ԛ??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE????St$????A[????<??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(\??????j+????AX9??v???Y?R?!?u??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?
F%u????????Aꕲq???Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[Ӽ???io???T??At$???~??YQ?|a??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??^)??ꕲq???A&S????Y䃞ͪϥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??o_??F%u???A|??Pk???Y?u?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,e?X???q??????A??????Y?H?}??*	33333g?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??K7?A??!?Q
?C@)???z6??1?/?6?B@:Preprocessing2F
Iterator::ModelD?l?????!H/???)@@)1?*????1?g?XNn3@:Preprocessing2U
Iterator::Model::ParallelMapV2&䃞ͪ??!8?z??)@)&䃞ͪ??18?z??)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??镲??!]蕚?P@)?w??#???1?ZV?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice c?ZB>??!??L?Wk@) c?ZB>??1??L?Wk@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate9??m4???!B??k:?)@)RI??&¶?1֛?>?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???ׁs??!??(??0@)>yX?5ͫ?1?˺?L@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???{????!??D~0@)???{????1??D~0@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??"?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$		|???S???Js?t????.n????!F%u???	!       "	!       *	!       2$	}1?˾B???JN??ߺ???????!|??Pk???:	!       B	!       J$	??ޡ??Z???p????Ɯ?!?Fx$??R	!       Z$	??ޡ??Z???p????Ɯ?!?Fx$??JCPU_ONLYY??"?@b Y      Y@q?O?%9@"?
both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?25.0279% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 